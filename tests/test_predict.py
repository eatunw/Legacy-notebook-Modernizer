"""
test_predict.py — pytest suite for the Titanic FastAPI /predict endpoint.

Categories
----------
1. Valid prediction tests  — checks business-logic probability thresholds.
2. Input validation tests  — malformed requests must return HTTP 422.
3. Schema tests            — response shape and field types/ranges.
4. Pipeline integrity test — identical inputs produce identical outputs (determinism).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# ---------------------------------------------------------------------------
# Shared TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """
    Spin up the app once for the entire module via TestClient.
    The lifespan context manager loads the pipeline at startup and tears it
    down after the last test, matching production startup/shutdown behaviour.
    """
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def post_predict(client: TestClient, payload: dict):
    return client.post("/predict", json=payload)


# ---------------------------------------------------------------------------
# 1. VALID PREDICTION TESTS
# ---------------------------------------------------------------------------

class TestValidPredictions:
    """End-to-end predictions against the serialised pipeline."""

    def test_male_3rd_class_low_survival(self, client):
        """Male, 3rd-class, SibSp=1, Parch=0 → historically low survival."""
        resp = post_predict(client, {"Pclass": 3, "Sex": "male", "SibSp": 1, "Parch": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["probability"] < 0.3, (
            f"Expected low survival probability (<0.3), got {body['probability']}"
        )

    def test_female_1st_class_high_survival(self, client):
        """Female, 1st-class, SibSp=0, Parch=0 → historically high survival."""
        resp = post_predict(client, {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert body["probability"] > 0.8, (
            f"Expected high survival probability (>0.8), got {body['probability']}"
        )

    def test_male_1st_class_middle_ground(self, client):
        """Male, 1st-class — middle-ground; just verify a valid prediction is returned."""
        resp = post_predict(client, {"Pclass": 1, "Sex": "male", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 200
        body = resp.json()
        # Probability must be a valid float in [0, 1]; survived must be 0 or 1.
        assert 0.0 <= body["probability"] <= 1.0
        assert body["survived"] in (0, 1)

    def test_female_3rd_class_returns_valid_prediction(self, client):
        """Female, 3rd-class, SibSp=0, Parch=2 — confirm the endpoint is reachable."""
        resp = post_predict(client, {"Pclass": 3, "Sex": "female", "SibSp": 0, "Parch": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["survived"] in (0, 1)


# ---------------------------------------------------------------------------
# 2. INPUT VALIDATION TESTS (expect HTTP 422)
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Pydantic should reject malformed inputs before they reach the model."""

    def test_missing_required_field_pclass(self, client):
        """Omitting Pclass must return 422."""
        resp = post_predict(client, {"Sex": "male", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 422

    def test_invalid_sex_value(self, client):
        """Sex='alien' is not 'male' or 'female' — must return 422."""
        resp = post_predict(client, {"Pclass": 2, "Sex": "alien", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 422

    def test_invalid_pclass_too_high(self, client):
        """Pclass=4 is outside the valid range [1, 3] — must return 422."""
        resp = post_predict(client, {"Pclass": 4, "Sex": "male", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 422

    def test_invalid_pclass_zero(self, client):
        """Pclass=0 is below the valid range — must return 422."""
        resp = post_predict(client, {"Pclass": 0, "Sex": "female", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 422

    def test_wrong_type_pclass_string(self, client):
        """Pclass='three' (string) — must return 422 (type coercion is not valid here)."""
        resp = post_predict(client, {"Pclass": "three", "Sex": "male", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 422

    def test_negative_sibsp(self, client):
        """SibSp=-1 violates the ge=0 constraint — must return 422."""
        resp = post_predict(client, {"Pclass": 1, "Sex": "male", "SibSp": -1, "Parch": 0})
        assert resp.status_code == 422

    def test_empty_body(self, client):
        """An empty JSON body must return 422."""
        resp = post_predict(client, {})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 3. SCHEMA TESTS
# ---------------------------------------------------------------------------

class TestResponseSchema:
    """The /predict response must contain exactly 'survived' and 'probability'."""

    def test_response_has_exactly_survived_and_probability(self, client):
        """Response JSON must have exactly the two documented keys."""
        resp = post_predict(client, {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"survived", "probability"}, (
            f"Unexpected response keys: {set(body.keys())}"
        )

    def test_survived_is_int_0_or_1(self, client):
        """'survived' must be an integer equal to 0 or 1."""
        resp = post_predict(client, {"Pclass": 3, "Sex": "male", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 200
        survived = resp.json()["survived"]
        assert isinstance(survived, int)
        assert survived in (0, 1)

    def test_probability_is_float_between_0_and_1(self, client):
        """'probability' must be a float in [0.0, 1.0]."""
        resp = post_predict(client, {"Pclass": 2, "Sex": "female", "SibSp": 1, "Parch": 1})
        assert resp.status_code == 200
        prob = resp.json()["probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_survived_consistent_with_probability(self, client):
        """'survived' must match the rounding of 'probability' at the 0.5 threshold."""
        resp = post_predict(client, {"Pclass": 1, "Sex": "male", "SibSp": 0, "Parch": 0})
        assert resp.status_code == 200
        body = resp.json()
        expected_label = 1 if body["probability"] >= 0.5 else 0
        assert body["survived"] == expected_label, (
            f"survived={body['survived']} inconsistent with probability={body['probability']}"
        )


# ---------------------------------------------------------------------------
# 4. PIPELINE INTEGRITY / DETERMINISM TEST
# ---------------------------------------------------------------------------

class TestPipelineIntegrity:
    """Calling /predict twice with the same payload must return identical results."""

    @pytest.mark.parametrize("payload", [
        {"Pclass": 3, "Sex": "male",   "SibSp": 1, "Parch": 0},
        {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0},
        {"Pclass": 2, "Sex": "male",   "SibSp": 0, "Parch": 1},
    ])
    def test_identical_inputs_produce_identical_outputs(self, client, payload):
        """Determinism check: two requests with the same body → same JSON response."""
        resp1 = post_predict(client, payload)
        resp2 = post_predict(client, payload)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json() == resp2.json(), (
            f"Non-deterministic output for payload {payload}:\n"
            f"  call 1 → {resp1.json()}\n"
            f"  call 2 → {resp2.json()}"
        )
