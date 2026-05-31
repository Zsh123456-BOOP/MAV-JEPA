from mavjepa.metrics import gsm8k_scores, normalize_number, normalize_sql, spider_scores, strip_sql_markdown


def test_gsm8k_number_normalization():
    assert normalize_number("$1,234.") == "1234"
    assert normalize_number("answer is 12.50") == "12.5"


def test_gsm8k_numeric_exact_match():
    scores = gsm8k_scores("Reasoning #### $1,234.", "#### 1234")
    assert scores["final_answer_exact_match"] is False
    assert scores["numeric_exact_match"] is True


def test_spider_sql_normalization_and_fence_strip():
    assert strip_sql_markdown("```sql\nSELECT * FROM t;\n```") == "SELECT * FROM t;"
    assert normalize_sql("SELECT  *\nFROM t;") == "select * from t"


def test_spider_string_exact_without_db():
    scores = spider_scores("```sql\nselect name from singer;\n```", "SELECT name FROM singer")
    assert scores["sql_string_exact_match"] is True
    assert scores["execution_accuracy"] is None
