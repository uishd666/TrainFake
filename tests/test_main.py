import argparse
import random
import subprocess
import sys
import unittest

from simulate_train import main as slack_main


def make_args(**overrides):
    defaults = {
        "steps": 10,
        "loss_start": 1.5,
        "loss_min": 0.1,
        "acc_start": 0.5,
        "oscillation": 0.05,
        "step_delay": 0,
        "log_every": 2,
        "log_style": "trainer",
        "scenario": "normal",
        "chaos_level": 1,
        "seed": 42,
        "save_every": 0,
        "save_delay": 0,
        "project_name": "project-name",
        "run_name": "run-name",
        "speed_jitter": 0.22,
        "rainbow": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class ValidationTests(unittest.TestCase):
    def test_valid_args_pass(self):
        slack_main.validate_args(make_args())

    def test_invalid_scenario_fails(self):
        with self.assertRaisesRegex(ValueError, "--scenario"):
            slack_main.validate_args(make_args(scenario="coffee-break"))

    def test_invalid_chaos_level_fails(self):
        with self.assertRaisesRegex(ValueError, "--chaos-level"):
            slack_main.validate_args(make_args(chaos_level=7))


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


class CliSmokeTests(unittest.TestCase):
    def run_cli(self, *args):
        command = [sys.executable, "-m", "simulate_train.main", *args]
        return subprocess.run(command, text=True, capture_output=True, check=True)

    def test_basic_cli_smoke(self):
        result = self.run_cli("--steps", "3", "--step-delay", "0", "--save-every", "0", "--log-every", "1")
        self.assertIn("'loss'", result.stdout)

    def test_llm_pretrain_chaos_smoke(self):
        result = self.run_cli(
            "--steps",
            "3",
            "--scenario",
            "llm-pretrain",
            "--chaos-level",
            "2",
            "--seed",
            "42",
            "--step-delay",
            "0",
            "--save-every",
            "0",
        )
        self.assertRegex(result.stdout, "WARNING|ERROR|INFO")

    def test_diffusion_style_smoke(self):
        result = self.run_cli(
            "--steps",
            "3",
            "--log-style",
            "stable-diffusion",
            "--scenario",
            "diffusion",
            "--step-delay",
            "0",
            "--save-every",
            "0",
        )
        self.assertIn("step_loss", result.stdout)


if __name__ == "__main__":
    unittest.main()
