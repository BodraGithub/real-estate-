"""
Real Estate Investment Advisor - Streamlit app (with machine learning).

1) python train.py            -> cleans data, trains models, logs to MLflow, saves ./models and ./outputs
2) streamlit run app.py       -> opens the app
"""
import io
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏠", layout="wide")

OUT, MODELS, PPSF, BLUE, CURRENT_YEAR = "outputs", "models", "Price_per_SqFt_calc", "#2a6fdb", 2025

import train as T   # cleaning + model training code (same folder)

# ------------------------------------------------------------------ data source: sidebar upload or saved default
DEFAULT_FILE = "india_housing_prices.csv"
FIT_ROWS = 80_000

st.sidebar.title("🏠 Investment Advisor")
st.sidebar.subheader("📁 Data")

upload = st.sidebar.file_uploader(
    "Upload india_housing_prices.csv (optional)",
    type="csv",
    help=f"Same columns as {DEFAULT_FILE}. If not uploaded, the app uses the pre-loaded "
         f"'{DEFAULT_FILE}' and the saved models.",
)


@st.cache_resource(show_spinner="Cleaning data and training models… this can take a few minutes for large files.")
def train_uploaded(data: bytes):
    raw = pd.read_csv(io.BytesIO(data))
    missing = [c for c in T.REQUIRED if c not in raw.columns]
    if missing:
        return {"missing": missing}
    return T.train_models(raw, fit_rows=FIT_ROWS)


@st.cache_resource
def load_saved():
    cleaned_path = f"{OUT}/india_housing_cleaned.csv"
    meta_path = f"{MODELS}/app_metadata.json"
    if not (os.path.exists(cleaned_path) and os.path.exists(meta_path)):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return dict(
        df=pd.read_csv(cleaned_path),
        meta=meta,
        clf=joblib.load(f"{MODELS}/best_classifier.joblib"),
        reg=joblib.load(f"{MODELS}/best_regressor.joblib"),
        clf_df=pd.read_csv(f"{OUT}/classification_results.csv").set_index("model"),
        reg_df=pd.read_csv(f"{OUT}/regression_results.csv").set_index("model"),
    )


@st.cache_resource(show_spinner="No saved models found — training from the local CSV…")
def load_default_csv():
    """Fallback: train from the local CSV if saved artifacts are missing."""
    if not os.path.exists(DEFAULT_FILE):
        return None
    raw = pd.read_csv(DEFAULT_FILE)
    missing = [c for c in T.REQUIRED if c not in raw.columns]
    if missing:
        return None
    return T.train_models(raw, fit_rows=FIT_ROWS)


# ---- pick the data source: uploaded file > saved artifacts > raw default CSV ----
if upload is not None:
    art = train_uploaded(upload.getvalue())
    if "missing" in art:
        st.title("🏠 Real Estate Investment Advisor")
        st.error("The uploaded file is missing these columns: " + ", ".join(art["missing"]))
        st.stop()
    st.sidebar.success(f"Using **{upload.name}**  \n{len(art['df']):,} rows after cleaning")
else:
    art = load_saved()
    if art is not None:
        st.sidebar.info(f"Using pre-loaded data: **{DEFAULT_FILE}**")
    else:
        art = load_default_csv()
        if art is None:
            st.title("🏠 Real Estate Investment Advisor")
            st.info(
                f"👈 Upload `{DEFAULT_FILE}` in the sidebar, or place it next to this app "
                "and run `python train.py` first to generate `outputs/` and `models/`."
            )
            st.stop()
        st.sidebar.success(f"Using pre-loaded data: **{DEFAULT_FILE}**")

df, clf, reg, meta = art["df"], art["clf"], art["reg"], art["meta"]
YEARS = int(meta["assumptions"]["years"])
if upload is not None and meta.get("trained_on_rows", 0) < len(df):
    st.sidebar.caption(f"Models trained on a {meta['trained_on_rows']:,}-row sample for speed.")


def fmt_lakh(x):
    return f"₹{x:,.1f} L" if x < 100 else f"₹{x / 100:,.2f} Cr"


# ------------------------------------------------------------------ EDA helpers (each returns fig, caption)
def _bar(d, x, y, title, order=None, agg="mean", ylabel="₹ per sq ft"):
    g = d.groupby(x)[y].agg(agg)
    if order:
        g = g.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g.index.astype(str), g.values, color=BLUE)
    ax.set(title=title, ylabel=ylabel, xlabel="")
    gap = (g.max() / g.min() - 1) * 100 if g.min() > 0 else float("nan")
    return fig, f"Highest vs lowest group differ by {gap:.1f}% (axis starts at zero)."


def _hist(col, title, xlabel):
    def f(d):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(d[col], bins=50, kde=True, ax=ax, color=BLUE)
        ax.set(title=title, xlabel=xlabel)
        return fig, f"Mean {d[col].mean():,.1f} | median {d[col].median():,.1f} | min {d[col].min():,.1f} | max {d[col].max():,.1f}"
    return f


def q3(d):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=d, x="Property_Type", y=PPSF, showfliers=False, color="#9ec1ff", ax=ax)
    ax.set(title="Price per sq ft by property type", ylabel="₹ per sq ft", xlabel="")
    return fig, "Median ₹/sq ft: " + ", ".join(f"{k} {v:,.0f}" for k, v in d.groupby("Property_Type")[PPSF].median().items())


def q4(d):
    s = d.sample(min(5000, len(d)), random_state=1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(s["Size_in_SqFt"], s["Price_in_Lakhs"], s=6, alpha=0.3)
    ax[0].set(title="Size vs price", xlabel="sq ft", ylabel="₹ lakhs")
    ax[1].scatter(s["Size_in_SqFt"], s[PPSF], s=6, alpha=0.3, color="#d9534f")
    ax[1].set(title="Size vs price per sq ft", xlabel="sq ft", ylabel="₹ per sq ft", yscale="log")
    return fig, f"Correlation size–price: {d['Size_in_SqFt'].corr(d['Price_in_Lakhs']):.3f}; size–₹/sq ft: {d['Size_in_SqFt'].corr(d[PPSF]):.3f}"


def q5(d):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, c, t in zip(ax, [PPSF, "Size_in_SqFt", "Price_in_Lakhs"], ["₹ per sq ft", "Size (sq ft)", "Price (₹ lakhs)"]):
        sns.boxplot(y=d[c], ax=a, color="#f0ad4e")
        a.set(title=t, ylabel="")
    return fig, f"{int(d['PPSF_Outlier_Flag'].sum()):,} rows ({d['PPSF_Outlier_Flag'].mean():.1%}) are price-per-sq-ft outliers (IQR rule); they are flagged, not deleted."


def _hbar(g, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(g))))
    ax.barh(g.index.astype(str), g.values, color=BLUE)
    ax.set(title=title, xlabel=xlabel)
    return fig, f"Highest: {g.index[-1]} ({g.values[-1]:,.0f}) | lowest: {g.index[0]} ({g.values[0]:,.0f})"


def q6(d):
    return _hbar(d.groupby("State")[PPSF].mean().sort_values(), "Average price per sq ft by state", "₹ per sq ft")


def q7(d):
    return _hbar(d.groupby("City")["Price_in_Lakhs"].mean().sort_values(), "Average property price by city", "₹ lakhs")


def q8(d):
    age = d.groupby("Locality")["Age_of_Property"].median()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(age, bins=20, ax=ax[0], color="#8e44ad")
    ax[0].set(title="Median age across localities", xlabel="years")
    ext = pd.concat([age.sort_values().head(5), age.sort_values().tail(5)])
    ax[1].barh(ext.index, ext.values, color="#8e44ad")
    ax[1].set(title="5 youngest and 5 oldest localities", xlabel="median age (years)")
    return fig, f"Median of the locality medians: {age.median():.1f} years."


def q9(d):
    t = pd.crosstab(d["City"], d["BHK"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(t))))
    sns.heatmap(t, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "% of city's listings"})
    ax.set(title="BHK distribution across cities (%)", ylabel="")
    return fig, ""


def q10(d):
    top5 = d.groupby("Locality")["Price_in_Lakhs"].mean().nlargest(5).index
    tr = (d[d["Locality"].isin(top5)].groupby(["Year_Built", "Locality"])["Price_in_Lakhs"].mean()
          .unstack().rolling(5, min_periods=1, center=True).mean())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tr.plot(ax=ax, marker="o", ms=3)
    ax.set(title="Price by year built – 5 most expensive localities (5-yr rolling mean)", xlabel="Year built", ylabel="₹ lakhs")
    return fig, "The data has no price history, so year built is used as the time axis."


def q11(d):
    cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", PPSF, "Age_of_Property", "Floor_No", "Total_Floors", "Nearby_Schools",
            "Nearby_Hospitals", "Parking_Space", "Security", "Amenities_Count", "Infra_Score"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(d[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
    ax.set(title="Correlation of numeric features")
    return fig, ""


def _line(col, title):
    def f(d):
        g = d.groupby(col)[PPSF].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(g.index, g.values, marker="o", color=BLUE)
        ax.set(title=title, xlabel=col, ylabel="₹ per sq ft")
        return fig, f"Correlation with ₹/sq ft: {d[col].corr(d[PPSF]):.3f}"
    return f


def q18(d):
    return _bar(d.assign(P=d["Parking_Space"].map({0: "No parking", 1: "Parking"})), "P", "Price_in_Lakhs",
                "Average price by parking space", ylabel="₹ lakhs")


def q19(d):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    g = d.groupby("Amenities_Count")[PPSF].mean()
    ax[0].bar(g.index.astype(str), g.values, color="#e67e22")
    ax[0].set(title="₹/sq ft by number of amenities", xlabel="amenities", ylabel="₹ per sq ft")
    eff = pd.Series({a: (d.loc[d[f"Has_{a}"] == 1, PPSF].mean() / d.loc[d[f"Has_{a}"] == 0, PPSF].mean() - 1) * 100
                     for a in ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]}).sort_values()
    ax[1].barh(eff.index, eff.values, color=np.where(eff.values >= 0, "#27ae60", "#c0392b"))
    ax[1].set(title="₹/sq ft difference when amenity present (%)", xlabel="%")
    return fig, ""


def q20(d):
    order = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, y, t in zip(ax, [PPSF, "Good_Investment"], ["₹ per sq ft", "Share of good investments"]):
        g = d.groupby("Public_Transport_Accessibility")[y].mean().reindex(order)
        a.bar(order, g.values, color="#8e44ad")
        a.set(title=t + " by transport access")
    return fig, "Good-investment share is driven by the rule-based label, which uses transport in its growth formula."


def x_area(d):
    bands = pd.cut(d["Size_in_SqFt"], 5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    d.groupby(bands, observed=True)["Return_5Y_Pct"].mean().plot(kind="bar", ax=ax[0], color="#16a085", rot=20)
    ax[0].set(title="Size band vs estimated 5-yr return (rule-based)", xlabel="", ylabel="%")
    d.groupby(bands, observed=True)[PPSF].mean().plot(kind="bar", ax=ax[1], color="#2980b9", rot=20)
    ax[1].set(title="Size band vs actual ₹/sq ft", xlabel="", ylabel="₹ per sq ft")
    return fig, f"corr(size, est. return) = {d['Size_in_SqFt'].corr(d['Return_5Y_Pct']):.3f}"


def x_infra(d):
    bands = pd.cut(d["Infra_Score"], 6)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    d.groupby(bands, observed=True)["Price_in_Lakhs"].mean().plot(kind="bar", ax=ax[0], color="#2980b9", rot=25)
    ax[0].set(title="Infrastructure score vs current price", xlabel="", ylabel="₹ lakhs")
    d.groupby(bands, observed=True)["Future_Price_5Y"].mean().plot(kind="bar", ax=ax[1], color="#e67e22", rot=25)
    ax[1].set(title=f"Infrastructure score vs est. {YEARS}-yr price (rule-based)", xlabel="", ylabel="₹ lakhs")
    return fig, "The right chart rises by construction because infrastructure is part of the growth rule."


GROUPS = {
    "1–5 Price & size": [
        ("Q1 · Distribution of property prices", _hist("Price_in_Lakhs", "Property price", "₹ lakhs")),
        ("Q2 · Distribution of property sizes", _hist("Size_in_SqFt", "Property size", "sq ft")),
        ("Q3 · Price per sq ft by property type", q3), ("Q4 · Size vs price", q4), ("Q5 · Outliers", q5)],
    "6–10 Location": [
        ("Q6 · Average price per sq ft by state", q6), ("Q7 · Average price by city", q7),
        ("Q8 · Median age by locality", q8), ("Q9 · BHK distribution across cities", q9),
        ("Q10 · Price trend – top 5 most expensive localities", q10)],
    "11–15 Relationships": [
        ("Q11 · Correlation of numeric features", q11),
        ("Q12 · Nearby schools vs price per sq ft", _line("Nearby_Schools", "Nearby schools vs avg ₹/sq ft")),
        ("Q13 · Nearby hospitals vs price per sq ft", _line("Nearby_Hospitals", "Nearby hospitals vs avg ₹/sq ft")),
        ("Q14 · Price by furnished status", lambda d: _bar(d, "Furnished_Status", "Price_in_Lakhs", "Average price by furnished status", ylabel="₹ lakhs")),
        ("Q15 · Price per sq ft by facing", lambda d: _bar(d, "Facing", PPSF, "Average ₹/sq ft by facing direction"))],
    "16–20 Ownership & amenities": [
        ("Q16 · Properties by owner type", lambda d: _bar(d, "Owner_Type", "ID", "Properties by owner type", agg="count", ylabel="properties")),
        ("Q17 · Properties by availability status", lambda d: _bar(d, "Availability_Status", "ID", "Properties by availability", agg="count", ylabel="properties")),
        ("Q18 · Effect of parking on price", q18), ("Q19 · Amenities vs price per sq ft", q19),
        ("Q20 · Public transport vs price / investment potential", q20)],
    "Extra analyses": [("Area vs investment return", x_area), ("Infrastructure score vs resale value", x_infra)],
}


# ------------------------------------------------------------------ sidebar
page = st.sidebar.radio("Go to", ["Overview", "Data & Features", "EDA (20 questions)", "Property Explorer",
                                  "Investment Advisor (ML)", "Model Performance"])
st.sidebar.caption(f"Classifier: {meta['best_classifier']}  \nRegressor: {meta['best_regressor']}")


# ------------------------------------------------------------------ pages
def page_overview():
    st.title("🏠 Real Estate Investment Advisor")
    st.caption("Predicting property profitability & future value with machine learning.")
    c = st.columns(4)
    c[0].metric("Properties", f"{len(df):,}")
    c[1].metric("Cities / States", f"{df['City'].nunique()} / {df['State'].nunique()}")
    c[2].metric("Good investments", f"{df['Good_Investment'].mean():.1%}")
    c[3].metric(f"Avg est. {YEARS}-yr return", f"{df['Return_5Y_Pct'].mean():.1f}%")
    st.markdown(f"""
**Two ML tasks**
1. **Classification** – is this property a *Good Investment*? (`{meta['best_classifier']}`)
2. **Regression** – estimated price after {YEARS} years. (`{meta['best_regressor']}`)

**Pages:** data cleaning & features · 20 EDA questions · property explorer with filters · the advisor form
(prediction + confidence + feature importance) · model comparison tracked with MLflow.
""")
    st.info("The dataset has no 'good investment' label and no future price, so both targets were **engineered from "
            "business rules** (see *Data & Features*). Model scores show how well the models learn those rules, "
            "not how real markets will behave.")
    st.dataframe(df.head(20))


def page_data():
    st.title("🧹 Data & Features")
    c = st.columns(4)
    c[0].metric("Rows (raw → clean)", f"{meta['rows_raw']:,} → {meta['rows_clean']:,}")
    c[1].metric("Columns (raw → clean)", f"{meta['cols_raw']} → {meta['cols_clean']}")
    c[2].metric("Missing values (raw)", f"{meta['missing_raw']:,}")
    c[3].metric("Duplicates removed", f"{meta['duplicates']:,}")
    st.markdown(f"""
**Handling steps**
* **Missing values / duplicates:** median / mode imputation and duplicate removal (ignoring `ID`).
* **Invalid floors:** {meta['floor_inconsistent']:,} rows have `Floor_No > Total_Floors`; they are flagged and `Floor_No` is capped.
* **Outliers:** {meta['ppsf_outliers']:,} price-per-sq-ft outliers (IQR rule) are flagged, not deleted.
* **Price per sq ft** is recomputed in ₹ from price and size; **age** is recomputed from `Year_Built`.
* **Encoding & scaling (inside the model pipeline):** one-hot encoding for categories, standard scaling for numbers.
""")
    st.subheader("Engineered features")
    st.table(pd.DataFrame([
        ("Amenities_Count, Has_*", "number of amenities and one flag per amenity"),
        ("School / Hospital_Density_Score", "min-max scaled nearby schools / hospitals (0–1)"),
        ("Transport_Score", "Low 0.33, Medium 0.67, High 1.0"),
        ("Infra_Score", "0.35 × school + 0.35 × hospital + 0.30 × transport"),
        ("Floor_Ratio, Size_per_BHK", "floor / total floors, size / BHK"),
        ("Price_vs_City_Median", "price per sq ft divided by the city's median"),
    ], columns=["Feature", "Definition"]).set_index("Feature"))
    a = meta["assumptions"]
    st.subheader("Target rules")
    st.markdown(f"""
`growth %/yr = {a['base_growth']} + {a['w_infra']}·Infra_Score + {a['w_under_construction']}·[Under construction]
+ {a['w_security']}·Security + {a['w_parking']}·Parking {a['w_age']:+}·Age`

`Future_Price_{YEARS}Y = Price × (1 + growth/100)^{YEARS}`

**Good Investment** = price per sq ft ≤ city median **and** growth ≥ {meta['growth_threshold']:.2f} %/yr (dataset median).
""")
    st.download_button("⬇️ Download cleaned dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="india_housing_cleaned.csv", mime="text/csv")
    st.dataframe(df.head(100))


def page_eda():
    st.title("📊 Exploratory Data Analysis")
    group = st.radio("Question group", list(GROUPS), horizontal=True)
    for title, fn in GROUPS[group]:
        st.subheader(title)
        fig, caption = fn(df)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if caption:
            st.caption(caption)
    if group == "Extra analyses":
        st.warning("**Crime rate vs Good Investment** cannot be analysed: the dataset has no crime-rate column.")


def page_explorer():
    st.title("🔎 Property Explorer")
    with st.expander("Filters", expanded=True):
        c1, c2, c3 = st.columns(3)
        states = c1.multiselect("State", sorted(df["State"].unique()))
        pool = sorted(df[df["State"].isin(states)]["City"].unique()) if states else sorted(df["City"].unique())
        cities = c2.multiselect("City", pool)
        types = c3.multiselect("Property type", sorted(df["Property_Type"].unique()))
        c4, c5 = st.columns(2)
        bhks = c4.multiselect("BHK", sorted(df["BHK"].unique()))
        status = c5.multiselect("Availability", sorted(df["Availability_Status"].unique()))
        c6, c7 = st.columns(2)
        pmin, pmax = float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())
        smin, smax = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].max())
        price = c6.slider("Price (₹ lakhs)", pmin, pmax, (pmin, pmax))
        size = c7.slider("Size (sq ft)", smin, smax, (smin, smax))
        only_good = st.checkbox("Only 'Good Investment' properties")
    m = df["Price_in_Lakhs"].between(*price) & df["Size_in_SqFt"].between(*size)
    for col, sel in [("State", states), ("City", cities), ("Property_Type", types), ("BHK", bhks), ("Availability_Status", status)]:
        if sel:
            m &= df[col].isin(sel)
    if only_good:
        m &= df["Good_Investment"] == 1
    f = df[m]
    if f.empty:
        st.warning("No properties match these filters.")
        return
    c = st.columns(4)
    c[0].metric("Matching", f"{len(f):,}")
    c[1].metric("Avg price", fmt_lakh(f["Price_in_Lakhs"].mean()))
    c[2].metric("Avg ₹/sq ft", f"₹{f[PPSF].mean():,.0f}")
    c[3].metric("Good investments", f"{f['Good_Investment'].mean():.1%}")
    show = ["ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs", PPSF,
            "Availability_Status", "Future_Price_5Y", "Return_5Y_Pct", "Good_Investment"]
    sort_by = st.selectbox("Sort by", ["Return_5Y_Pct", "Price_in_Lakhs", PPSF, "Size_in_SqFt"])
    st.dataframe(f.sort_values(sort_by, ascending=st.checkbox("Ascending"))[show].head(500).round(2))
    st.caption("First 500 rows of the sorted result.")
    st.download_button("⬇️ Download filtered rows (CSV)", f[show].to_csv(index=False).encode("utf-8"),
                       file_name="filtered_properties.csv", mime="text/csv")
    g1, g2 = st.columns(2)
    with g1:
        top = f.groupby("City")["Price_in_Lakhs"].mean().sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.barh(top.index, top.values, color=BLUE)
        ax.set(title="Avg price by city (top 15)", xlabel="₹ lakhs")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with g2:
        heat = f.pivot_table(index="State", columns="Property_Type", values="Good_Investment", aggfunc="mean") * 100
        fig, ax = plt.subplots(figsize=(6, 4.2))
        sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGn", ax=ax, cbar_kws={"label": "% good"})
        ax.set(title="Good-investment share (%) by state × type", xlabel="", ylabel="")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    t = f.groupby("Year_Built")["Price_in_Lakhs"].mean().rolling(3, min_periods=1, center=True).mean()
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(t.index, t.values, color=BLUE)
    ax.set(title="Avg price by year built (3-yr rolling)", xlabel="Year built", ylabel="₹ lakhs")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


def importance_chart(imp, title):
    s = pd.Series(imp).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(s.index, s.values, color=BLUE)
    ax.set(title=title, xlabel="drop in score when shuffled")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)


def page_advisor():
    st.title("💡 Investment Advisor")
    st.caption(f"Enter a property. The trained models predict whether it is a good investment and its price in {YEARS} years.")
    cat = meta["categories"]
    with st.form("advisor"):
        c1, c2, c3 = st.columns(3)
        city = c1.selectbox("City", sorted(meta["city_to_state"]))
        ptype = c2.selectbox("Property type", cat["Property_Type"])
        bhk = c3.number_input("BHK", 1, 10, 3)
        c4, c5, c6 = st.columns(3)
        size = c4.number_input("Size (sq ft)", 100, 20000, int(np.clip(1500, *meta["size_range"])), step=50)
        price = c5.number_input("Price (₹ lakhs)", 1.0, 5000.0, float(np.clip(250.0, *meta["price_range"])), step=5.0)
        year = c6.number_input("Year built", 1900, CURRENT_YEAR, int(np.clip(2007, meta["year_range"][0], CURRENT_YEAR)))
        c7, c8, c9 = st.columns(3)
        total_floors = c7.number_input("Total floors", 1, 100, 10)
        floor_no = c8.number_input("Floor number", 0, 100, 3)
        furnished = c9.selectbox("Furnished status", cat["Furnished_Status"])
        c10, c11, c12 = st.columns(3)
        schools = c10.number_input("Nearby schools", 0, 100, max(1, meta["school_max"] // 2))
        hospitals = c11.number_input("Nearby hospitals", 0, 100, max(1, meta["hosp_max"] // 2))
        transport = c12.selectbox("Public transport accessibility", cat["Public_Transport_Accessibility"])
        c13, c14, c15 = st.columns(3)
        parking = c13.checkbox("Parking space", value=True)
        security = c14.checkbox("Security", value=True)
        amenities = c15.multiselect("Amenities", ["Gym", "Pool", "Clubhouse", "Playground", "Garden"], default=["Gym", "Pool"])
        c16, c17, c18 = st.columns(3)
        facing = c16.selectbox("Facing", cat["Facing"])
        owner = c17.selectbox("Owner type", cat["Owner_Type"])
        status = c18.selectbox("Availability status", cat["Availability_Status"])
        go = st.form_submit_button("Evaluate property")
    if not go:
        st.info("Fill in the form and press **Evaluate property**.")
        return

    floor_no = min(floor_no, total_floors)
    row = {"City": city, "Property_Type": ptype, "Furnished_Status": furnished, "Public_Transport_Accessibility": transport,
           "Facing": facing, "Owner_Type": owner, "Availability_Status": status, "BHK": bhk, "Size_in_SqFt": size,
           "Price_in_Lakhs": price, "Price_per_SqFt_calc": price * 1e5 / size, "Floor_No": floor_no,
           "Total_Floors": total_floors, "Floor_Ratio": floor_no / total_floors, "Age_of_Property": CURRENT_YEAR - year,
           "Nearby_Schools": schools, "Nearby_Hospitals": hospitals, "Parking_Space": int(parking),
           "Security": int(security), "Amenities_Count": len(amenities), "Size_per_BHK": size / bhk}
    X = pd.DataFrame([row])[meta["features"]]
    p_good = float(clf.predict_proba(X)[0, 1])
    good = p_good >= 0.5
    future = float(reg.predict(X)[0])
    ret = (future / price - 1) * 100
    cagr = ((future / price) ** (1 / YEARS) - 1) * 100

    st.subheader("Result")
    if good:
        st.success("✅ **Good Investment**")
    else:
        st.error("❌ **Not a Good Investment**")
    k = st.columns(4)
    k[0].metric("Price today", fmt_lakh(price))
    k[1].metric(f"Est. price in {YEARS} yrs", fmt_lakh(future), f"{ret:.1f}%")
    k[2].metric("Implied growth", f"{cagr:.2f}%/yr")
    k[3].metric("Model confidence", f"{max(p_good, 1 - p_good):.1%}")
    st.progress(min(max(p_good, 0.0), 1.0), text=f"Probability of 'Good Investment': {p_good:.1%}")

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(range(YEARS + 1), [price * (1 + cagr / 100) ** t for t in range(YEARS + 1)], marker="o", color=BLUE)
        ax.set(title="Estimated price path", xlabel="Years from now", ylabel="₹ lakhs")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with g2:
        importance_chart(meta["classifier_importance"], "What drives the classifier (feature importance)")

    st.subheader("Compared with similar listings")
    sim = df[(df["City"] == city) & (df["BHK"] == bhk) & df["Price_in_Lakhs"].between(price * 0.8, price * 1.2)]
        if sim.empty:
        st.caption("No similar listings (same city, same BHK, ±20% price).")
    else:
        s = st.columns(3)
        s[0].metric("Similar listings", f"{len(sim):,}")
        s[1].metric("Avg ₹/sq ft", f"₹{sim[PPSF].mean():,.0f}")
        s[2].metric("Good-investment share", f"{sim['Good_Investment'].mean():.1%}")
        st.dataframe(sim.sort_values("Return_5Y_Pct", ascending=False)[
            ["ID", "Locality", "Property_Type", "Size_in_SqFt", "Price_in_Lakhs", "Future_Price_5Y", "Good_Investment"]].head(10).round(2))
    st.caption("Predictions come from models trained on rule-based targets. They are not real market forecasts or financial advice.")


def page_models():
    st.title("📈 Model Performance")
    cm, rm = meta["classifier_metrics"], meta["regressor_metrics"]
    st.subheader(f"Classification – {meta['best_classifier']}")
    c = st.columns(5)
    for col, name in zip(c, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(name.replace("_", " ").upper(), f"{cm[name]:.3f}")
    st.caption(f"Baseline (always predict the majority class): accuracy {meta['baseline_accuracy']:.3f}")
    st.dataframe(art["clf_df"].round(4))
    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        sns.heatmap(np.array(meta["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"])
        ax.set(title="Confusion matrix (test set)", xlabel="Predicted", ylabel="Actual")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with r:
        importance_chart(meta["classifier_importance"], "Classifier feature importance")

    st.subheader(f"Regression – {meta['best_regressor']}")
    c = st.columns(3)
    c[0].metric("RMSE (₹ lakhs)", f"{rm['rmse']:.2f}")
    c[1].metric("MAE (₹ lakhs)", f"{rm['mae']:.2f}")
    c[2].metric("R²", f"{rm['r2']:.4f}")
    st.dataframe(art["reg_df"].round(4))
    importance_chart(meta["regressor_importance"], "Regressor feature importance")
    st.markdown("**MLflow:** every model run is logged. View it with  \n"
                "`mlflow ui --backend-store-uri sqlite:///mlflow.db`")
    st.warning("Scores are high because the targets were created from rules on the same data. Compare with the baseline "
               "and do not read them as real-market accuracy.")


{"Overview": page_overview, "Data & Features": page_data, "EDA (20 questions)": page_eda,
 "Property Explorer": page_explorer, "Investment Advisor (ML)": page_advisor,
 "Model Performance": page_models}[page]()  
