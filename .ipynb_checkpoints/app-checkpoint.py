import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="EduPro Analytics Platform",
    layout="wide"
)

# Platform Header
st.title("📚 EduPro Analytics Intelligence Platform")
st.markdown("""
### Predictive Modeling for Course Demand and Revenue Forecasting

Forecast enrollments, estimate revenue potential, and explore key business insights across EduPro's course portfolio.
""")
st.divider()

# -------------------------
# Load Data & Models
# -------------------------
@st.cache_data
def get_clean_data():
    return pd.read_csv("data/model_data.csv")

df = get_clean_data()

enrollment_model = joblib.load("models/enrollment_model.pkl")
revenue_model = joblib.load("models/revenue_model.pkl")

# -------------------------
# Sidebar Navigation
# -------------------------
page = st.sidebar.selectbox(
    "Navigation",
    [
        "Overview",
        "Enrollment Prediction",
        "Revenue Prediction",
        "EDA Insights",
        "Business Insights"
    ]
)

# -------------------------
# Overview Page
# -------------------------
if page == "Overview":
    st.header("Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📚 Total Courses", f"{len(df)}")
        
    with col2:
        st.metric("🏷️ Categories", f"{df['CourseCategory'].nunique()}")
        
    with col3:
        st.metric("💰 Platform Revenue", f"${df['TotalRevenue'].sum():,.0f}")

# -------------------------
# EDA Insights Page (Graph moved to the top!)
# -------------------------
elif page == "EDA Insights":
    st.header("🔍 Comprehensive Exploratory Data Analysis (EDA)")
    st.markdown("---")

    # Visual Elements Placed First
    st.subheader("📊 Interactive Revenue Visualizations")
    revenue_by_category = df.groupby("CourseCategory")["TotalRevenue"].sum().sort_values(ascending=False)
    st.bar_chart(revenue_by_category)
    
    top_category = revenue_by_category.idxmax()
    st.success(f"🏆 Top Performing Category: {top_category}")
    
    st.subheader("📋 Raw Category Performance Breakdown")
    category_table = (
        revenue_by_category.reset_index()
        .rename(columns={"CourseCategory": "Category", "TotalRevenue": "Total Revenue ($)"})
    )
    st.dataframe(category_table, use_container_width=True, hide_index=True)

    st.divider()

    # SECTION 1 & 2: Top Drivers & Portfolio Mix
    st.subheader("1. Category Revenue Dominance & Portfolio Mix")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        #### 🚀 1. Artificial Intelligence is the Top Revenue Category
        * **AI Revenue:** `$202,750.67`
        * AI generates the absolute highest single-category revenue on the platform.
        
        👉 **Recommendation:** Increase investment in core technical frontiers including **Deep Learning**, **Computer Vision**, and **Generative AI** as they carry the strongest commercial potential.
        """)
        
    with col_b:
        st.markdown("""
        #### 📈 2. Business & Project Management are Strong Secondary Drivers
        * **Combined Revenue:** `$181,527.58` (Business) + `$169,103.21` (PM) = **`$350,630.79`**
        * Combined, these non-technical fields comfortably exceed the AI category alone.
        
        👉 **Recommendation:** Continue expanding tracks like **Business Strategy** and **Leadership**. The platform's health is safely diversified beyond pure programming tracks.
        """)

    st.markdown("---")

    # SECTION 3 & 4: Concentration & Audience Level
    st.subheader("2. Revenue Concentration & Course Levels")
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("""
        #### ⚠️ 3. Revenue Concentration & Lagging Segments
        * **Machine Learning:** `$133.38`
        * **Web Development:** `$16,682.15`
        * **Marketing:** `$0.00`
        
        👉 **Recommendation:** Immediate strategic intervention required. Conduct targeted product audits inspecting **course quality, pricing structures, promotions, and platform positioning** for these lagging buckets.
        """)
        
    with col_d:
        st.markdown("""
        #### 🎓 4. Beginner Courses Attract the Highest Revenue
        * **Beginner Avg Revenue:** `$17,572.45`
        * **Intermediate Avg Revenue:** `$14,681.44`
        * **Advanced Avg Revenue:** `$13,239.82`
        
        💡 *Business Interpretation:* Beginner courses generate roughly **~33% more revenue** than highly advanced courses.
        
        👉 **Recommendation:** Aggressively scale beginner-level configurations; they possess the lowest barriers to entry and capture broad customer registration pools.
        """)

    st.markdown("---")

    # SECTION 5, 6, 7 & 8: Correlation & Platform Characteristics
    st.subheader("3. Behavioral Trajectories & Demand Elasticity")
    col_e, col_f = st.columns(2)
    
    with col_e:
        st.markdown("""
        #### ⭐ 5. Course Ratings Positively Affect Demand
        * **Correlation ($CourseRating \\leftrightarrow EnrollmentCount$):** `+0.29`
        * This represents one of the strongest, highly actionable relationships uncovered.
        
        👉 **Recommendation:** Direct resources into improving existing core content and enhancing student support vectors to pull up baseline scores.
        
        #### 💵 6. Price Slightly Elasticizes Enrollment
        * **Correlation ($CoursePrice \\leftrightarrow EnrollmentCount$):** `-0.16`
        * The relationship is predictably negative but remarkably weak.
        
        👉 **Recommendation:** Premium pricing models remain safe to execute; steep price targets will not completely cannibalize demand.
        """)
        
    with col_f:
        st.markdown("""
        #### ⚖️ 7. Enrollment Distribution is Exceptionally Stable
        * **Platform Mean Enrollment:** `166.67` students
        * **Platform Median Enrollment:** `166` students
        * **Standard Deviation:** `12.52`
        
        💡 *Interpretation:* Demand flows evenly. The ecosystem doesn't suffer from extreme, hyper-skewed "superstar course" variances.
        
        #### 👨‍🏫 8. Instructor Experience Carries Zero Impact
        * **Correlation ($YearsOfExperience \\leftrightarrow EnrollmentCount$):** `0.006`
        
        👉 **Recommendation:** Students heavily prioritize syllabus alignment, course ratings, and direct topic relevance far over an instructor's decade-long corporate portfolio.
        """)

    # CRITICAL MODELING WARNING BLOCK
    st.error("### 🛑 Critical Preprocessing Modeling Observation")
    st.markdown("""
    During diagnostic exploratory modeling, perfect linear dependencies were isolated:
    * $\\text{CoursePrice} \\leftrightarrow \\text{TotalRevenue} = 1.00$
    * $\\text{CoursePrice} \\leftrightarrow \\text{AvgRevenue} = 1.00$
    * $\\text{EnrollmentCount} \\leftrightarrow \\text{UniqueStudents} = 1.00$
    
    **Data Preprocessing Pipeline Action:** To fully mitigate severe multicollinearity and feature redundancy prior to model fitting, **`AvgRevenue`** and **`UniqueStudents`** will be dropped systematically from the final training arrays.
    """)

    st.info("""
    ### 🏁 EDA Conclusion
    Platform revenue is structurally calculated via **Course Category, Course Price, Course Rating, and Course Level**, while human-capital variables such as **Teacher Experience and Teacher Rating** show minor to no impact on customer enrollment decisions.
    """)

# -------------------------
# Enrollment Prediction Page
# -------------------------
elif page == "Enrollment Prediction":
    st.header("🎯 Enrollment Demand Prediction")
    
    left, right = st.columns(2)
    
    with left:
        category = st.selectbox("Course Category", sorted(df["CourseCategory"].unique()))
        course_type = st.selectbox("Course Type", sorted(df["CourseType"].unique()))
        level = st.selectbox("Course Level", sorted(df["CourseLevel"].unique()))
        expertise = st.selectbox("Instructor Expertise", sorted(df["Expertise"].unique()))
        
    with right:
        price = st.number_input("Course Price ($)", min_value=0.0, value=100.0)
        duration = st.number_input("Duration (Hours)", min_value=1.0, value=20.0)
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0)
        experience = st.number_input("Instructor Experience (Years)", min_value=0, value=5)
        
    st.markdown("---")
    if st.button("Predict Enrollment"):
        sample = pd.DataFrame({
            "CourseCategory": [category], "CourseType": [course_type], "CourseLevel": [level],
            "CoursePrice": [price], "CourseDuration": [duration], "CourseRating": [rating],
            "Expertise": [expertise], "YearsOfExperience": [experience], "PriceBand": ["Medium"],
            "DurationBucket": ["Medium"], "RatingTier": ["Medium"], "ExperienceBucket": ["Mid"],
            "ExpertiseMatch": [0], "LevelEncoded": [1]
        })
        
        prediction = enrollment_model.predict(sample)
        st.metric("🎯 Expected Enrollments", f"{prediction[0]:.0f} Students")
        st.info("Estimated demand based on course attributes and historical platform behavior.")

# -------------------------
# Revenue Prediction Page
# -------------------------
elif page == "Revenue Prediction":
    st.header("💰 Revenue Forecasting")
    
    left, right = st.columns(2)
    
    with left:
        category = st.selectbox("Course Category", sorted(df["CourseCategory"].unique()), key="rev_cat")
        course_type = st.selectbox("Course Type", sorted(df["CourseType"].unique()), key="rev_type")
        level = st.selectbox("Course Level", sorted(df["CourseLevel"].unique()), key="rev_level")
        expertise = st.selectbox("Instructor Expertise", sorted(df["Expertise"].unique()), key="rev_exp")
        
    with right:
        price = st.number_input("Course Price ($)", min_value=0.0, value=100.0, key="rev_price")
        duration = st.number_input("Duration (Hours)", min_value=1.0, value=20.0, key="rev_duration")
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0, key="rev_rating")
        experience = st.number_input("Instructor Experience (Years)", min_value=0, value=5, key="rev_yrs")
        
    st.markdown("---")
    if st.button("Predict Revenue"):
        sample = pd.DataFrame({
            "CourseCategory": [category], "CourseType": [course_type], "CourseLevel": [level],
            "CoursePrice": [price], "CourseDuration": [duration], "CourseRating": [rating],
            "Expertise": [expertise], "YearsOfExperience": [experience], "PriceBand": ["Medium"],
            "DurationBucket": ["Medium"], "RatingTier": ["Medium"], "ExperienceBucket": ["Mid"],
            "ExpertiseMatch": [0], "LevelEncoded": [1]
        })
        
        prediction = revenue_model.predict(sample)
        st.metric("💰 Expected Revenue", f"${prediction[0]:,.0f}")
        st.info("Projected course revenue generated from the predictive forecasting model.")

# -------------------------
# Strategic Business Insights Page
# -------------------------
elif page == "Business Insights":
    st.header("📊 Strategic Insights")
    
    st.markdown("""
    ### Key Findings From Empirical Modeling
    
    #### 🚀 Highest Revenue Categories
    - **Artificial Intelligence** (Platform leading revenue contributor)
    - **Business**
    - **Project Management**
    
    #### 🎓 Beginner Courses Perform Best
    Introductory and beginner-level entry points demonstrate structural consistency, pulling the highest average validation revenue metrics across the platform.
    
    #### ⭐ Ratings Drive Demand
    Course validation evaluation metrics isolate clear positive linear trajectories balancing historical `CourseRating` patterns directly against high registration counts.
    
    #### 💵 Pricing Drives Revenue
    Feature mapping structures prove that raw baseline parameters governed by `CoursePrice` hold strong directional linear behavior directly computing revenue models.
    
    #### 👨‍🏫 Instructor Experience Has Limited Impact
    Statistical predictive behaviors confirm customer bases choose domain specificity, course structure, and material presentation over the length of instructor tenure.
    """)