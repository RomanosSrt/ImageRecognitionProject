# Data Preprocessing Overview

## Folder: DataPreProcess

The `DataPreProcess` folder contains several operational scripts designed to clean, transform, and prepare raw data for analysis. Below is a summary of the key steps and processes involved in these scripts:

### 1. Data Cleaning
- **Script:** `clean_data.py`
- **Description:** This script removes any missing or duplicate values from the dataset. It also handles outliers and corrects any inconsistencies in the data.

### 2. Data Transformation
- **Script:** `transform_data.py`
- **Description:** This script applies various transformations to the data, such as normalization, scaling, and encoding categorical variables. It ensures that the data is in a suitable format for analysis or modeling.

### 3. Feature Engineering
- **Script:** `feature_engineering.py`
- **Description:** This script creates new features from the existing data to improve the performance of machine learning models. It includes techniques like polynomial features, interaction terms, and domain-specific feature creation.

### 4. Data Splitting
- **Script:** `split_data.py`
- **Description:** This script splits the dataset into training, validation, and test sets. It ensures that the data is divided in a way that maintains the integrity of the analysis and prevents data leakage.

### 5. Data Export
- **Script:** `export_data.py`
- **Description:** This script exports the cleaned and processed data to a specified format (e.g., CSV, JSON) for further analysis or use in machine learning models.

## Summary
The scripts in the `DataPreProcess` folder are essential for preparing raw data for analysis. They ensure that the data is clean, transformed, and ready for use in various analytical and modeling tasks.





As viewed per the freequency histogram of ratings per user, the dataset does not follow a canonical distribution. Selected where users with 5 or more ratings and 100 and less ratings to keep the set not biased.