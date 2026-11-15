import textwrap

from streamlit.testing.v1 import AppTest

import src.research_validation as research_validation


def _write_research_app(tmp_path):
    app_path = tmp_path / "research_validation_app.py"
    app_path.write_text(
        textwrap.dedent(
            """
            import numpy as np
            import src.research_validation as research_validation

            research_validation.run_sensitivity_analysis_tab(
                ["Capaya"], 1, 4, 2, 0.05, 0.8, 1000, None,
                np.zeros((2, 2)), {"Capaya": 1}, {"Capaya": 100}, 30,
            )
            research_validation.render_research_baseline_comparison(
                ["Capaya"], None, np.zeros((2, 2)), {"Capaya": 1},
                {
                    "total": 10.0,
                    "waste_dict": {"Capaya": 100},
                    "capacity": 1000,
                    "params": {"n_vehicles": 1},
                    "routes": [["Capaya"]],
                    "feasible": False,
                },
            )
            """
        ),
        encoding="utf-8",
    )
    return app_path


def test_sensitivity_table_shows_each_seed_feasibility_and_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        research_validation,
        "run_benchmark",
        lambda *args, **kwargs: {
            "seeds": [30, 31, 32, 33, 34],
            "totals": [12.0, 11.0, 14.0, 10.0, 13.0],
            "mean_total": 12.0,
            "std_total": 1.4,
            "feasible": [True, False, True, False, True],
            "timed_out": [False, True, False, False, True],
        },
    )
    monkeypatch.setattr(
        research_validation, "evaluate_manual_baseline", lambda *args, **kwargs: (20.0, [["Capaya"]])
    )
    monkeypatch.setattr(
        research_validation, "greedy_baseline", lambda *args, **kwargs: (15.0, [["Capaya"]])
    )

    app = AppTest.from_file(str(_write_research_app(tmp_path))).run()
    assert not app.exception

    app.button[0].click().run()

    assert not app.exception
    sensitivity_table = next(
        element.value
        for element in app.dataframe
        if "Seed" in element.value.columns
    )
    assert list(sensitivity_table.columns) == [
        "Seed",
        "Total Distance (km)",
        "Feasible",
        "Timed Out",
    ]
    assert sensitivity_table.to_dict(orient="records") == [
        {"Seed": 30, "Total Distance (km)": 12.0, "Feasible": True, "Timed Out": False},
        {"Seed": 31, "Total Distance (km)": 11.0, "Feasible": False, "Timed Out": True},
        {"Seed": 32, "Total Distance (km)": 14.0, "Feasible": True, "Timed Out": False},
        {"Seed": 33, "Total Distance (km)": 10.0, "Feasible": False, "Timed Out": False},
        {"Seed": 34, "Total Distance (km)": 13.0, "Feasible": True, "Timed Out": True},
    ]


def test_comparison_uses_ga_feasibility_when_total_capacity_is_sufficient(monkeypatch, tmp_path):
    monkeypatch.setattr(
        research_validation,
        "evaluate_manual_baseline",
        lambda *args, **kwargs: (20.0, [["Capaya"]]),
    )
    monkeypatch.setattr(
        research_validation,
        "greedy_baseline",
        lambda *args, **kwargs: (15.0, [["Capaya"]]),
    )

    app = AppTest.from_file(str(_write_research_app(tmp_path))).run()

    assert not app.exception
    markdown = [element.value for element in app.markdown]
    assert any("GA Diagnostic" in value for value in markdown)
    assert not any("GA Optimized" in value for value in markdown)
    assert any("infeasible diagnostic" in warning.value for warning in app.warning)
    comparison_table = next(
        element.value for element in app.dataframe if "Method" in element.value.columns
    )
    ga_result = comparison_table.loc[comparison_table["Method"] == "GA"].iloc[0]
    assert ga_result["Feasibility"] == "Infeasible diagnostic"


def _write_statistical_app(tmp_path):
    app_path = tmp_path / "statistical_verification_app.py"
    app_path.write_text(
        textwrap.dedent(
            """
            import numpy as np
            from src.research_validation import run_statistical_verification
            run_statistical_verification(
                2, ["Capaya"], 1, 4, 2, 0.05, 0.8, 1000, None,
                np.array([[0.0, 1.0], [1.0, 0.0]]), {"Capaya": 1},
                {"Capaya": 100}, base_seed=40,
            )
            """
        ),
        encoding="utf-8",
    )
    return app_path


def test_statistical_verification_suppresses_claims_for_unfeasible_seed(monkeypatch, tmp_path):
    statuses = iter([{"feasible": True}, {"feasible": False}])
    monkeypatch.setattr(
        research_validation, "validate_ga_inputs", lambda *args, **kwargs: {"Capaya": 100}
    )
    monkeypatch.setattr(
        research_validation,
        "evaluate_manual_baseline",
        lambda *args, **kwargs: (20.0, [["Capaya"]]),
    )
    monkeypatch.setattr(research_validation, "build_seed_chromosomes", lambda *args, **kwargs: [])

    def fake_run_ga(*args, **kwargs):
        kwargs["run_status"].update(next(statuses))
        return [["Capaya"]], [10.0], 10.0, []

    monkeypatch.setattr(research_validation, "run_ga", fake_run_ga)
    app = AppTest.from_file(str(_write_statistical_app(tmp_path))).run()

    assert not app.exception
    assert not any("Significant" in element.value for element in app.success)
    assert any("infeasible or unverified" in element.value for element in app.warning)
    table = app.dataframe[0].value
    assert list(table.columns) == ["Seed", "GA Distance", "Feasible", "Baseline"]
    assert table["Feasible"].tolist() == [True, False]


def test_statistical_verification_keeps_inference_for_fully_feasible_seeds(monkeypatch, tmp_path):
    monkeypatch.setattr(
        research_validation, "validate_ga_inputs", lambda *args, **kwargs: {"Capaya": 100}
    )
    monkeypatch.setattr(
        research_validation,
        "evaluate_manual_baseline",
        lambda *args, **kwargs: (20.0, [["Capaya"]]),
    )
    monkeypatch.setattr(research_validation, "build_seed_chromosomes", lambda *args, **kwargs: [])

    def fake_run_ga(*args, **kwargs):
        kwargs["run_status"].update({"feasible": True})
        return [["Capaya"]], [10.0], 10.0, []

    monkeypatch.setattr(research_validation, "run_ga", fake_run_ga)
    app = AppTest.from_file(str(_write_statistical_app(tmp_path))).run()

    assert not app.exception
    assert app.success
    assert any("Results Statement" in element.value for element in app.info)
