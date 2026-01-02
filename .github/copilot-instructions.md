# AI Coding Agent Instructions

## Project Overview
This workspace contains a collection of mini machine learning projects in Python, primarily focused on data science and predictive modeling. The main project is a house price prediction model using scikit-learn, implemented in Jupyter notebooks.

## Architecture & Data Flow
- **Data Loading**: CSV files loaded from `mini projects/data.csv` using pandas
- **Preprocessing Pipeline**: 
  - Numerical features: SimpleImputer (mean) + StandardScaler
  - Ordinal categorical: SimpleImputer (most_frequent) + OrdinalEncoder
  - Nominal categorical: SimpleImputer (most_frequent) + OneHotEncoder
- **Feature Engineering**: Custom features like `housing_age`, `totalsf`, `totalarea`, `totalbaths`, `totalporchsf`
- **Target Transformation**: Log transformation of `SalePrice` using `np.log1p`
- **Modeling**: LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor

## Key Workflows
- **EDA**: Use matplotlib/seaborn for scatter plots, histograms; identify outliers with z-scores and manual inspection
- **Data Cleaning**: Fill missing values based on domain knowledge (e.g., 'Unf' for basement finish type)
- **Model Training**: sklearn Pipeline with ColumnTransformer for preprocessing + model
- **Evaluation**: Mean squared error on test set

## Project-Specific Conventions
- **Imports**: Group by purpose (data manipulation, ML, visualization) with comments
- **Data Handling**: Separate train/test DataFrames from start; apply transformations to both
- **Feature Selection**: Drop highly correlated features (e.g., keep GarageCars, drop GarageArea)
- **Outlier Removal**: Manual identification via scatter plots and domain knowledge
- **Encoding**: OrdinalEncoder for ordered categories (e.g., quality scales), OneHotEncoder for unordered
- **Pipelines**: Use sklearn Pipeline and ColumnTransformer for reproducible preprocessing

## Common Patterns
- Check `housing_data.duplicated().sum()` and `housing_data.isnull().sum()` early
- Visualize relationships with `plt.scatter()` and `sns.catplot()` for categorical
- Correlation heatmap with `sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')`
- Split data with `train_test_split(test_size=0.2, random_state=42)`

## Key Files
- `mini projects/house_pricepred 1.ipynb`: Main house price prediction notebook
- `mini projects/data.csv`: Housing dataset (Ames Housing)
- `mini projects/heart disease/`: Another mini project folder
- `mini projects/PRCP-1028-Skin-Disorder-Prediction/`: Skin disorder prediction project

## Dependencies
- pandas, numpy, scipy, sklearn, matplotlib, seaborn
- Jupyter notebooks for interactive development

## Validation
After changes, run relevant cells to ensure preprocessing and model training work without errors.