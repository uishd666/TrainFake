import io
import random
import subprocess
import sys
import unittest
from pathlib import Path

from rich.console import Console

from simulate_train import main as slack_main


ROOT = Path(__file__).resolve().parent.parent


class SpecTests(unittest.TestCase):
    def test_builtin_preset_resolves(self):
        args = slack_main.parse_args(["run", "llama-70b-pretrain", "--step-delay", "0"])
        spec = slack_main.resolve_run_spec(args)
        self.assertEqual(spec.name, "llama-70b-pretrain")
        self.assertEqual(spec.step_delay, 0)

    def test_unknown_preset_fails_clearly(self):
        args = slack_main.parse_args(["run", "missing-preset"])
        with self.assertRaisesRegex(ValueError, "unknown preset"):
            slack_main.resolve_run_spec(args)

    def test_yaml_config_loads_with_defaults(self):
        spec = slack_main.load_run_spec(str(ROOT / "tests" / "fixtures" / "sample_run.yaml"))
        self.assertEqual(spec.name, "fixture-run")
        self.assertEqual(spec.log_style, "trainer")
        self.assertEqual(spec.steps, 8)
        self.assertEqual(spec.save_every, 0)
        self.assertEqual(len(spec.stages), 3)

    def test_invalid_stage_range_fails(self):
        spec = slack_main.RunSpec(
            name="bad",
            description="bad",
            stages=(slack_main.StageSpec("train", 0.6, 0.4),),
        )
        with self.assertRaisesRegex(ValueError, "start_pct"):
            slack_main.validate_run_spec(spec)

    def test_new_playful_presets_resolve(self):
        for preset in (
            "rlhf-reward-model",
            "rag-eval-nightly",
            "multimodal-pretrain",
            "k8s-gpu-drill",
        ):
            with self.subTest(preset=preset):
                args = slack_main.parse_args(["run", preset, "--step-delay", "0", "--steps", "5"])
                spec = slack_main.resolve_run_spec(args)
                slack_main.validate_run_spec(spec)
                self.assertEqual(spec.name, preset)


class FormattingTests(unittest.TestCase):
    def test_checkpoint_path(self):
        self.assertEqual(
            slack_main.checkpoint_path("demo", "run-a", 1900),
            "demo/run-a/checkpoint-1900/checkpoint.pth",
        )

    def test_zero_delay_stays_zero(self):
        rng = random.Random(1)
        self.assertEqual(slack_main.simulated_step_delay(1, 0, 0.5, rng), 0)

    def test_trainer_log_contains_expected_fields(self):
        metrics = slack_main.simulated_metrics(1, 3, 1.5, 0.1, 0.5, 0.05, rng=random.Random(1))
        log = slack_main.format_log("trainer", 1, 3, metrics, False)
        self.assertIn("'loss'", log)
        self.assertIn("'grad_norm'", log)
        self.assertIn("'learning_rate'", log)
        self.assertIn("'epoch'", log)

    def test_new_scenario_logs_contain_signature_fields(self):
        cases = {
            "rlhf": "reward_accuracy",
            "rag-eval": "faithfulness",
            "multimodal": "contrastive_loss",
            "k8s-gpu": "pod",
        }
        for scenario, expected in cases.items():
            with self.subTest(scenario=scenario):
                metrics = slack_main.simulated_metrics(
                    3,
                    5,
                    1.5,
                    0.1,
                    0.5,
                    0.05,
                    scenario=scenario,
                    rng=random.Random(1),
                )
                log = slack_main.format_log("trainer", 3, 5, metrics, False, scenario)
                self.assertIn(expected, log)


class EventTests(unittest.TestCase):
    def event_sequence(self, seed):
        rng = random.Random(seed)
        events = []
        for step in range(1, 6):
            metrics = slack_main.simulated_metrics(
                step,
                5,
                1.5,
                0.1,
                0.5,
                0.05,
                scenario="llm-pretrain",
                rng=rng,
            )
            event = slack_main.training_event(step, 5, "llm-pretrain", 2, metrics, rng)
            if event:
                events.append(event)
        return events

    def test_seed_reproducible_events(self):
        self.assertEqual(self.event_sequence(42), self.event_sequence(42))

    def test_different_seeds_can_change_events(self):
        self.assertNotEqual(self.event_sequence(1), self.event_sequence(2))

    def test_chaos_zero_disables_events(self):
        metrics = slack_main.simulated_metrics(1, 3, 1.5, 0.1, 0.5, 0.05, rng=random.Random(1))
        event = slack_main.training_event(1, 3, "normal", 0, metrics, random.Random(1))
        self.assertIsNone(event)

    def test_new_scenario_events_are_seed_reproducible(self):
        for scenario in ("rlhf", "rag-eval", "multimodal", "k8s-gpu"):
            with self.subTest(scenario=scenario):
                metrics = slack_main.simulated_metrics(
                    2,
                    5,
                    1.5,
                    0.1,
                    0.5,
                    0.05,
                    scenario=scenario,
                    rng=random.Random(7),
                )
                first = slack_main.training_event(2, 5, scenario, 2, metrics, random.Random(7))
                second = slack_main.training_event(2, 5, scenario, 2, metrics, random.Random(7))
                self.assertEqual(first, second)
                self.assertIsNotNone(first)


class CinematicRunTests(unittest.TestCase):
    def test_summary_contains_core_counts(self):
        spec = slack_main.RunSpec(
            name="unit-run",
            description="unit",
            steps=6,
            step_delay=0,
            save_every=0,
            log_every=2,
            stages=slack_main.default_stages("finetune", 0),
        )
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        summary = slack_main.run_cinematic(spec, console)
        text = output.getvalue()
        self.assertEqual(summary.steps, 6)
        self.assertIsNotNone(summary.final_metrics)
        self.assertIn("summary: unit-run", text)
        self.assertIn("final loss", text)

    def test_presets_output_contains_professional_and_parody(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        slack_main.list_presets(console)
        text = output.getvalue()
        self.assertIn("llama-70b-pretrain", text)
        self.assertIn("boss-is-watching", text)
        self.assertIn("rlhf-reward-model", text)
        self.assertIn("rag-eval-nightly", text)
        self.assertIn("multimodal-pretrain", text)
        self.assertIn("k8s-gpu-drill", text)


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, *args):
        command = [sys.executable, "-m", "simulate_train.main", *args]
        return subprocess.run(command, text=True, capture_output=True, check=True)

    def test_presets_cli_smoke(self):
        result = self.run_cli("presets")
        self.assertIn("llama-70b-pretrain", result.stdout)
        self.assertIn("boss-is-watching", result.stdout)

    def test_builtin_run_cli_smoke(self):
        result = self.run_cli("run", "boss-is-watching", "--step-delay", "0", "--steps", "5", "--save-every", "0")
        self.assertIn("TrainFake run: boss-is-watching", result.stdout)
        self.assertIn("summary: boss-is-watching", result.stdout)

    def test_config_run_cli_smoke(self):
        result = self.run_cli(
            "run",
            "--config",
            str(ROOT / "tests" / "fixtures" / "sample_run.yaml"),
            "--step-delay",
            "0",
        )
        self.assertIn("TrainFake run: fixture-run", result.stdout)
        self.assertIn("summary: fixture-run", result.stdout)

    def test_new_preset_cli_smoke(self):
        for preset in (
            "rlhf-reward-model",
            "rag-eval-nightly",
            "multimodal-pretrain",
            "k8s-gpu-drill",
        ):
            with self.subTest(preset=preset):
                result = self.run_cli("run", preset, "--step-delay", "0", "--steps", "5", "--save-every", "0")
                self.assertIn(f"TrainFake run: {preset}", result.stdout)
                self.assertIn(f"summary: {preset}", result.stdout)

    def test_legacy_flags_print_migration_hint(self):
        command = [sys.executable, "-m", "simulate_train.main", "--steps", "3"]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("TrainFake uses a preset-first CLI", result.stdout)
        self.assertIn("trainfake run", result.stdout)


if __name__ == "__main__":
    unittest.main()
