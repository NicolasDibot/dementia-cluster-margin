"""Variable identifiers and roles used in Supplementary Table S2."""

from .geometry import FUNCTION_PROFILE_ITEMS

_HRS = (
    ("phase","","supplementary","n","wave","analysis wave factor"),
    ("ragender","ragender","supplementary","n","sex","RAND HRS ragender"),
    ("newage","agey_m","supplementary","s","age","wave-specific age at interview midpoint"),
    ("raeducl","raeduc","supplementary","n","education","1 if raeduc=1; 2 if raeduc in 2,3; 3 if raeduc in 4,5"),
    ("hlthlma","hlthlm","active","n","function_qualitative","health problems limit work"),
    ("walkra","walkra","active","n","function_qualitative","difficulty walking across a room"),
    ("dressa","dressa","active","n","function_qualitative","difficulty dressing"),
    ("batha","batha","active","n","function_qualitative","difficulty bathing or showering"),
    ("eata","eata","active","n","function_qualitative","difficulty eating"),
    ("beda","beda","active","n","function_qualitative","difficulty getting in or out of bed"),
    ("toilta","toilta","active","n","function_qualitative","difficulty using the toilet"),
    ("phonea","phonea","active","n","function_qualitative","difficulty using the telephone"),
    ("medsa","medsa","active","n","function_qualitative","difficulty taking medication"),
    ("moneya","moneya","active","n","function_qualitative","difficulty managing money"),
    ("shopa","shopa","active","n","function_qualitative","difficulty shopping for groceries"),
    ("mealsa","mealsa","active","n","function_qualitative","difficulty preparing a hot meal"),
    ("mapa","mapa","active","n","function_qualitative","difficulty using a map"),
    ("housewka","limhwrk","active","n","function_qualitative","Harmonized HRS: whether health limits housework"),
    ("walk100a","walk1a","active","n","function_qualitative","difficulty walking one block"),
    ("sita","sita","active","n","function_qualitative","difficulty sitting for two hours"),
    ("chaira","chaira","active","n","function_qualitative","difficulty getting up from a chair"),
    ("climsa","climsa","active","n","function_qualitative","difficulty climbing several flights of stairs"),
    ("clim1a","clim1a","active","n","function_qualitative","difficulty climbing one flight of stairs"),
    ("lifta","lifta","active","n","function_qualitative","difficulty lifting or carrying ten pounds"),
    ("stoopa","stoopa","active","n","function_qualitative","difficulty stooping kneeling or crouching"),
    ("armsa","armsa","active","n","function_qualitative","difficulty reaching or extending arms"),
    ("pusha","pusha","active","n","function_qualitative","difficulty pushing or pulling a large object"),
    ("dimea","dimea","active","n","function_qualitative","difficulty picking up a dime"),
    ("adltot_s","adltot6","active","s","function_quantitative","Harmonized HRS six-item ADL total"),
    ("iadltot1_s","iadltot_h","active","s","function_quantitative","Harmonized HRS six-item IADL total"),
    ("cogothp","proxy","external_descriptive","","measurement_process","whether the interview used a proxy"),
    ("imrc","imrc","active","s","cognitive_tests","immediate word recall"),
    ("dlrc","dlrc","active","s","cognitive_tests","delayed word recall"),
    ("tr20","tr20","active","s","cognitive_tests","total word recall"),
)


def variable_dictionary():
    fields = ("variable", "source_stem", "role", "mfa_type", "block", "measure")
    hrs = [dict(zip(fields, values)) for values in _HRS]
    share = []
    for original in hrs:
        row = dict(original)
        row["source_stem"] = row["variable"]
        if row["variable"] in FUNCTION_PROFILE_ITEMS:
            row["measure"] = "Harmonized SHARE released " + row["variable"]
        if row["variable"] == "cogothp":
            share.append(dict(zip(fields, ("cogimp", "cogimp", "active", "n",
                                          "cognitive_context", "Released cognitive-impairment indicator"))))
            row.update(role="active", mfa_type="n", block="cognitive_context",
                       measure="Released other-person response indicator")
        if row["variable"] == "newage":
            row["source_stem"] = "iwy, iwm, rabyear, rabmonth"
            row["measure"] = "Age calculated from interview and birth month/year"
        if row["variable"] == "raeducl":
            row["measure"] = "Released three-level education grouping"
        share.append(row)
    share.extend([
        dict(zip(fields, ("verbf", "verbf", "active", "s", "cognitive_tests", "Verbal fluency"))),
        dict(zip(fields, ("wtresp", "wtresp", "supplementary", "s", "weights", "Released respondent weight"))),
        dict(zip(fields, ("ipw_weight", "", "supplementary", "s", "weights", "Sequential stabilized attrition weight"))),
    ])
    if sum(row["role"] == "active" for row in hrs) != 29 or sum(row["role"] == "active" for row in share) != 32:
        raise RuntimeError("MFA variable counts disagree with the specified cohort groupings")
    return {"HRS": hrs, "SHARE": share,
            "notes": {"n": "categorical block", "s": "standardized quantitative block",
                      "cohort_adaptation": "Shared analysis names do not imply identical measurements across cohorts.",
                      "HRS_proxy": "Retained for eligibility, excluded from imputation and the active factor space.",
                      "SHARE_context": "The two cognitive-context indicators are active."}}
