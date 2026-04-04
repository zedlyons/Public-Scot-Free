from flask import Flask, render_template, request
from joblib import load
import polars as pl


app = Flask(__name__)
app.logger.info("Starting loading block")
model = load("scotfreerandomforestmodel.joblib")
app.logger.info("Model loaded")
agencies = pl.read_parquet("agencies.parquet")
app.logger.info("Parquet loaded")


# identity-infos of agencies are split across several columns so
# catenate those elements together into one name column
agencies = agencies.with_columns(
    pl.concat_str(
        [
            pl.col("ucr_agency_name"),
            pl.col("ncic_agency_name"),
            pl.col("pub_agency_name"),
            pl.col("pub_agency_unit"),
            pl.col("agency_type_name"),
        ],
        separator=" | ",
        ignore_nulls=True,
    ).alias("agency_full_name")
)


# Home page url takes user here. Show welcome info in home.html
@app.route("/", methods=["POST", "GET"])
def find_agency():
    if request.method == "GET":  # get user input for date and state of crime
        return render_template("home.html")
    else:  # "POST" user has given state/date data

        # user can now pick which geo/temporally-correct agency investigates
        state = request.form["state"]
        date_string = request.form["date"]
        date = pl.Series([date_string]).str.to_date()

        ag_year = date.dt.year()[0]
        # we assume that data is loaded in solid blocks of years
        if (ag_year > agencies["data_year"].unique()).all():
            ag_year = agencies["data_year"].max()
        elif (ag_year < agencies["data_year"].unique()).all():
            ag_year = agencies["data_year"].min()
        # want the agencies (and their associated pop stats) from closest year
        # if correct year is not available

        # make dataframe of agencies from correct year/state
        relevant_agen = agencies.filter(
            (pl.col("data_year") == ag_year) & (pl.col("state_abbr") == state)
        )

        return render_template(
            "agencies.html",
            datevar=date_string,
            statevar=state,
            agen_options=relevant_agen,
            ag_year=ag_year,
        )


@app.route("/prediction", methods=["POST", "GET"])
def predict():

    # ---begin variable request block---
    ag_name_string = request.form["choice"]
    ag_year = int(request.form["ag_year"])
    state = request.form["state"]
    date_str = request.form["date"]
    date = pl.Series([date_str]).str.to_date()
    sex = request.form["vic_sex"]
    vic_type = request.form["vic_type"]
    crim_act = request.form["gang"]
    prop_destruction = 1 if "prop_destruction" in request.form else 0
    taken_by_cops = 1 if "taken_by_cops" in request.form else 0
    looted_remains = 1 if "looted_remains" in request.form else 0
    feet_and_fists = 1 if "feet_and_fists" in request.form else 0
    vehicle_as_weapon = 1 if "vehicle_as_weapon" in request.form else 0
    blunt_weapon = 1 if "blunt_weapon" in request.form else 0
    weaponized_fire = 1 if "weaponized_fire" in request.form else 0
    choked_or_strangled = 1 if "choked_or_strangled" in request.form else 0
    poison = 1 if "poison" in request.form else 0
    other_weapon = 1 if "other_weapon" in request.form else 0
    num_of_victims = int(request.form["num_victims"]) + 1  # add 1 for primary victim
    num_of_killers = int(request.form["num_killers"])
    # ---end variable request block---

    # This df has all the agency-related data stored in one row
    agency_data = agencies.filter(
        (pl.col("data_year") == ag_year)
        & (pl.col("state_abbr") == state)
        & (pl.col("agency_full_name") == ag_name_string)
    )

    # many values are hard-coded. Tried to limit the user's ability to choose unknown-like options
    input_data = pl.DataFrame(
        {
            "incident_year": date.dt.year()[0],
            "incident_monthday": date.dt.day()[0],
            "incident_yearday": date.dt.ordinal_day()[0],
            "report_date_flag": 0,
            "state_abbr": state,
            "agency_type_name": agency_data["agency_type_name"][0],
            "population": agency_data["population"][0],
            "suburban_area_flag": agency_data["suburban_area_flag"][0],
            "population_group_desc": agency_data["population_group_desc"][0],
            "pop_sort_order": agency_data["pop_sort_order"][0],
            "male_officer": agency_data["male_officer"][0],
            "male_civilian": agency_data["male_civilian"][0],
            "male_total": agency_data["male_total"][0],
            "female_officer": agency_data["female_officer"][0],
            "female_civilian": agency_data["female_civilian"][0],
            "female_total": agency_data["female_total"][0],
            "officer_rate": agency_data["officer_rate"][0],
            "employee_rate": agency_data["employee_rate"][0],
            "county_name": agency_data["county_name"][0],
            "sex_code": sex,
            "age_range_high_num": None,
            "victim_age_status": "age_known",
            "victim_type_name": vic_type,
            "criminal_act_name": crim_act,
            "prop_destruction": prop_destruction,
            "taken_by_cops": taken_by_cops,
            "looted_remains": looted_remains,
            "feet_and_fists": feet_and_fists,
            "vehicle_as_weapon": vehicle_as_weapon,
            "blunt_weapon": blunt_weapon,
            "weaponized_fire": weaponized_fire,
            "weapons_unknown": 0,
            "choked_or_strangled": choked_or_strangled,
            "poison": poison,
            "other_weapon": other_weapon,
            "num_of_victims": num_of_victims,
            "num_of_killers": num_of_killers,
            "population_is_null": 1 if agency_data["population"].is_null()[0] else 0,
            "male_officer_is_null": (
                1 if agency_data["male_officer"].is_null()[0] else 0
            ),
            "male_civilian_is_null": (
                1 if agency_data["male_civilian"].is_null()[0] else 0
            ),
            "male_total_is_null": 1 if agency_data["male_total"].is_null()[0] else 0,
            "female_officer_is_null": (
                1 if agency_data["female_officer"].is_null()[0] else 0
            ),
            "female_civilian_is_null": (
                1 if agency_data["female_civilian"].is_null()[0] else 0
            ),
            "female_total_is_null": (
                1 if agency_data["female_total"].is_null()[0] else 0
            ),
            "officer_rate_is_null": (
                1 if agency_data["officer_rate"].is_null()[0] else 0
            ),
            "employee_rate_is_null": (
                1 if agency_data["employee_rate"].is_null()[0] else 0
            ),
            "criminal_act_name_is_null": 0,
        }
    )

    # feed user-given data into the model
    output = model.predict_proba(input_data)
    # model.classes_ returns array([0, 1], dtype=int32),
    # i.e. model's estimation output is of form:
    # [probability of getting caught, probability of getting away scot-free]

    return render_template(
        "prediction.html",
        scotfree_prob=str(round(output[0][1] * 100, 2)) + "%",
        statevar="prediction_gtg",
    )


if __name__ == "__main__":
    app.run(debug=True)
