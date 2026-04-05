from pipeline_test_runner import run_test_pipeline

def test_pipeline_runs():
    df = run_test_pipeline()

    assert len(df) > 0
    assert "authenticity_score" in df.columns
    assert df["authenticity_score"].between(0, 100).all()


def test_pipeline_no_nan():
    df = run_test_pipeline()

    assert df.isnull().sum().sum() == 0