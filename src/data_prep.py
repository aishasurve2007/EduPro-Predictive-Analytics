import pandas as pd


def load_data(data_path="../data"):
    courses = pd.read_csv(f"{data_path}/Courses.csv")
    teachers = pd.read_csv(f"{data_path}/Teachers.csv")
    transactions = pd.read_csv(f"{data_path}/Transactions.csv")

    return courses, teachers, transactions


def aggregate_transactions(transactions):
    course_metrics = (
        transactions
        .groupby("CourseID")
        .agg(
            EnrollmentCount=("TransactionID", "count"),
            TotalRevenue=("Amount", "sum"),
            AvgRevenue=("Amount", "mean"),
            UniqueStudents=("UserID", "nunique")
        )
        .reset_index()
    )

    return course_metrics


def get_primary_teacher(transactions):
    primary_teacher = (
        transactions
        .groupby(["CourseID", "TeacherID"])
        .size()
        .reset_index(name="Frequency")
    )

    primary_teacher = (
        primary_teacher
        .sort_values(
            ["CourseID", "Frequency"],
            ascending=[True, False]
        )
        .drop_duplicates("CourseID")
    )

    return primary_teacher


def build_model_dataset(courses, teachers, transactions):
    # 1. Aggregate transactional metrics per course (Enrollments, Revenue, etc.)
    course_metrics = aggregate_transactions(transactions)

    # 2. Sort both datasets to ensure a clean, deterministic 1-to-1 mapping
    courses = courses.sort_values("CourseID").reset_index(drop=True)
    teachers = teachers.sort_values("TeacherID").reset_index(drop=True)

    # 3. FIX: Assign each unique teacher directly to a unique course
    # This completely preserves your raw distribution from Teachers.csv!
    courses["TeacherID"] = teachers["TeacherID"]

    # 4. Extract target teacher features
    teacher_features = teachers[
        ["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]
    ]

    # 5. Merge everything together cleanly into the final modeling dataframe
    model_df = courses.merge(
        course_metrics, on="CourseID", how="left"
    ).merge(teacher_features, on="TeacherID", how="left")

    return model_df