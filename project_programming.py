import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ---Import the dataset---
wine_dataset = pd.read_csv('/Users/mac/Documents/1_Thạc sỹ/Programming and Database/Programming/WineDataset.csv')
wine_dataset.info()

# ---Page design---
st.set_page_config(page_title="Wine Produce Analysis", layout="wide")

#manual item selection
if st.session_state.get('switch_button', False):
    st.session_state['menu_option'] = (st.session_state.get('menu_option', 0) + 1) % 4
    manual_select = st.session_state['menu_option']
else:
    manual_select = None

#add on_change callback
def on_change(key):
    selection = st.session_state[key]
    st.write(f"Selection changed to {selection}")

#for easily change name of options 
Explore = '1. Explore the Dataset'
Clean = '2. Clean the Dataset'
Plots = '3. Interesting Plots'
Models = '4. Models Explain the Dataset'

#menu
with st.sidebar:
    selected = option_menu(
        menu_title='Menu', 
        options= [Explore, Clean, Plots, Models],
        default_index=0,
        manual_select=manual_select,
        on_change=on_change,
        key='menu_main',
        styles={
        'container': {'padding': '15px!important', 'background-color': '#d9d9d9'},
        'nav-link': {'font-size': '12px',  'color': 'black', 'text-align': 'left', 'margin':'0px', '--hover-color': '#8a8a8a'},
        'nav-link-selected': {'background-color': '#fafafa', 'color': 'black' , 'font-weight': 'normal'},
        'menu-title': {'font-size': '18px', 'font-weight': 'bold', 'color': 'black', 'text-align': 'left'}, 
        }
    ) 
    selected


# ---Explore the dataset---
if selected == Explore:
    st.header('Research about :blue[Wine Produce]', divider='grey')
    st.subheader(Explore)
    
    show_code_1 = '''
    wine_dataset = pd.read_csv(
    '/Users/mac/Documents/1_Thạc sỹ/Programming and Database/Programming/WineDataset.csv')
    wine_dataset.info()
    '''
    st.subheader('Import the Dataset')
    st.code(show_code_1, language='python')


    st.write('### Original Wine Dataset')
    st.write('1. Initial Dataset')
    #show initial dataset
    st.dataframe(wine_dataset)
    #check rows and columns
    rows = wine_dataset.shape[0]
    cols = wine_dataset.shape[1]
    st.write(f"Number of Rows: {rows}")
    st.write(f"Number of Columns: {cols}")
    
    st.write('2. Dataset Description')
    st.dataframe(wine_dataset.describe().T)
    
    #missing values check
    st.write('3. Check for missing values')
    st.dataframe(wine_dataset.isnull().sum())


    #take a look
    st.write('### Rows of the Dataset')
    st.write('1. First 6 rows')
    st.dataframe(wine_dataset.head(6))
    
    st.write('2. Last 6 Rows')
    st.dataframe(wine_dataset.tail(6))

    #count unique value of all columns
    st.write('3. Count Unique Value')
    st.dataframe(wine_dataset.nunique())


    st.write('### The Values of some Notable Columns')
    #per bottle/ case/ each value counts
    st.write('1. Per bottle / case / each Column')
    per_bottle_case_each_counts = wine_dataset['Per bottle / case / each'].value_counts()
    per_bottle_case_each_counts_df = per_bottle_case_each_counts.to_frame().T
    st.dataframe(per_bottle_case_each_counts_df)
    #plot a chart
    plt.figure(figsize=(8,3))
    plt.bar(per_bottle_case_each_counts.index, per_bottle_case_each_counts.values, color='blue')
    plt.xlabel('Per/bottle/case/each', fontsize = 6)
    plt.ylabel('Count', fontsize = 6)
    plt.xticks(fontsize = 6)
    plt.yticks(fontsize = 6)
    st.pyplot(plt)

    #type value counts
    st.write('2. Type Column')
    st.dataframe(wine_dataset['Type'].value_counts())
    
    #capacity value counts
    st.write('3. Capacity Column')
    capacity_counts =wine_dataset['Capacity'].value_counts()
    capacity_counts_df = capacity_counts.to_frame().T
    st.dataframe(capacity_counts_df)
    #plot a chart
    plt.figure(figsize=(8,3))
    plt.bar(capacity_counts.index, capacity_counts.values, color='skyblue')
    plt.xlabel('Capacity', fontsize = 6)
    plt.ylabel('Count', fontsize = 6)
    plt.xticks(rotation=45, fontsize = 6)
    plt.yticks(fontsize = 6)
    st.pyplot(plt)
    
    #style value counts
    st.write('4. Style Column')
    style_counts = wine_dataset['Style'].value_counts()
    style_counts_df = style_counts.to_frame().T
    st.dataframe(style_counts_df)
    #plot a chart
    plt.figure(figsize=(12, 8))
    plt.pie(style_counts, labels=style_counts.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    st.pyplot(plt)
    
    #region value counts
    st.write('5. Region Column')
    region_counts = wine_dataset['Region'].value_counts()
    region_counts_df = region_counts.to_frame().T
    st.dataframe(region_counts_df)
    #plot a chart
    plt.figure(figsize=(12, 8))
    plt.pie(region_counts, labels=region_counts.index, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    st.pyplot(plt)
    
    
# ---Clean the dataset---
#make a copy of a initial dataset
wine_dataset_clean = wine_dataset.copy()

#drop NA and duplicate values
wine_dataset_clean.dropna(inplace=True)
wine_dataset_clean.drop_duplicates(inplace=True)

#drop some unnecessary columns
wine_dataset_clean.drop(columns=['Title'], inplace=True)
wine_dataset_clean.drop(columns=['Description'], inplace=True)
wine_dataset_clean.drop(columns=['Per bottle / case / each'], inplace=True)
wine_dataset_clean.drop(columns=['Secondary Grape Varieties'], inplace=True)

#intensive clean the Vintage column (remove "NV" value, make sure it is in 20xx format)
wine_dataset_clean = wine_dataset_clean[wine_dataset_clean['Vintage'] != 'NV']
wine_dataset_clean = wine_dataset_clean[wine_dataset_clean['Vintage'].str.match(r'^20\d{2}$')]


#creat a function to extract number from the cell
def clean_column(dataset, column):
    #remove non-numeric characters, then convert to float
    dataset[column] = pd.to_numeric(dataset[column].str.extract('(\d+\.\d+)')[0], errors='coerce')
    return dataset

#extract number from Price and ABV column
wine_dataset_clean = clean_column(wine_dataset_clean, 'Price')
wine_dataset_clean = clean_column(wine_dataset_clean, 'ABV')


#streamlit code
if selected == Clean:
    st.header('Research about :blue[Wine Produce]', divider='grey')
    st.subheader(Clean)

    st.write('### Code are use for clean dataset')
    #make a copy of the dataset
    show_code_1 = '''wine_dataset_clean = wine_dataset.copy()'''
    st.write('1. Make a Copy of the Dataset')
    st.code(show_code_1, language='python')


    #remove missing values and duplicates
    show_code_2 = '''
    wine_dataset_clean.dropna(inplace=True)
    wine_dataset_clean.drop_duplicates(inplace=True)
    '''
    st.write('2. Remove NA and Duplicate Values')
    st.code(show_code_2, language='python')
    

    #remove some unnecessary columns
    show_code_3 = '''
    wine_dataset_clean.drop(columns=['Title'], inplace=True)
    wine_dataset_clean.drop(columns=['Description'], inplace=True)
    wine_dataset_clean.drop(columns=['Per bottle / case / each'], inplace=True)
    wine_dataset_clean.drop(columns=['Secondary Grape Varieties'], inplace=True)
    '''
    st.write('3. Drop some unnecessary columns')
    st.code(show_code_3, language='python')
    
    show_code_4 = '''
    wine_dataset_clean = wine_dataset_clean[wine_dataset_clean['Vintage'] != 'NV']
    wine_dataset_clean = wine_dataset_clean[wine_dataset_clean['Vintage'].str.match(r'^20\d{2}$')]
    '''
    st.write('4. Intensive clean the Vintage column')
    st.code(show_code_4, language='python')
    
    show_code_5 = '''
    def clean_column(df, column):
        # Remove non-numeric characters, then convert to float
        df[column] = pd.to_numeric(df[column].str.extract('(\d+\.\d+)')[0], errors='coerce')
        return df
    wine_dataset_clean = clean_column(wine_dataset_clean, 'Price')
    wine_dataset_clean = clean_column(wine_dataset_clean, 'ABV')
    '''
    st.write('5. A function to extract number from a string')
    st.code(show_code_5, language='python')

    show_code_6 = '''
    wine_dataset_clean = clean_column(wine_dataset_clean, 'Price')
    wine_dataset_clean = clean_column(wine_dataset_clean, 'ABV')
    '''
    st.write('6. Extract number from a string')
    st.code(show_code_6, language='python')
    
    #display the cleaned dataset
    st.write('### Cleaned Wine Dataset')
    st.write('1. New dataset')
    st.dataframe(wine_dataset_clean)
    clean_rows = wine_dataset_clean.shape[0]
    clean_cols = wine_dataset_clean.shape[1]
    st.write(f"Number of Rows after Cleaning: {clean_rows}")
    st.write(f"Number of Columns after Cleaning: {clean_cols}")

    #check information on the new dataset
    st.write('2. Last 6 Rows of the Cleaned Dataset')
    st.dataframe(wine_dataset_clean.tail(6))
    
    #max and min of ABV
    st.write('3. Maximum and Minimum of ABV in %')
    st.write(f"Maximum = {wine_dataset_clean.ABV.max()} %")
    st.write(f"Minimum = {wine_dataset_clean.ABV.min()} %")
    
    #describe the new price column
    st.write('4. Describe the New Price Column')
    st.dataframe(wine_dataset_clean['Price'].describe().T)
    st.write(f"Maximum = £{wine_dataset_clean.Price.max()}")
    st.write(f"Minimum = £{wine_dataset_clean.Price.min()}")
    
    #Show the histogram of the price of the bottle
    fig, ax = plt.subplots(figsize=(12, 6))
    # Create the histogram
    ax.hist(wine_dataset_clean['Price'], bins=120, color='skyblue', edgecolor='black')
    ax.set_title('Histogram of Price of Wine')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    st.pyplot(fig)
    


df = pd.DataFrame(wine_dataset_clean)
st.dataframe(df)

#visualization
if selected == Plots:
    st.header('Research about :blue[Wine produce]', divider='grey')
    st.subheader(Plots)


    st.write('1. Average Price of Wine Bottles by Vintage')
    avg_price_by_Vintage = df.groupby('Vintage')['Price'].mean()    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(avg_price_by_Vintage.index, avg_price_by_Vintage.values, marker='o', linestyle='-')
    ax.set_title('Average Price of Wine by Vintage')
    ax.set_xlabel('Vintage', fontsize=6)
    ax.set_ylabel('Average Bottle Price (£)', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True)
    st.pyplot(fig)
    st.dataframe(avg_price_by_Vintage)


    st.write('2. Distribution of Wine Styles by Country')
    country_style_counts = wine_dataset_clean.groupby(['Country', 'Style']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    country_style_counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title('Distribution of Wine Styles by Country')
    ax.set_xlabel('Country')
    ax.set_ylabel('Number of Bottles')
    ax.set_xticks(range(len(country_style_counts.index)))
    ax.set_xticklabels(country_style_counts.index, rotation=45)
    ax.legend(title='Wine Style', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(country_style_counts)


    st.write('3. Count of Each Grape by Country')
    grape_by_country = df.groupby(['Grape', 'Country']).size().reset_index(name='number_of_bottle')
    plt.figure(figsize=(8, 6))
    plt.bar(grape_by_country['Country'] + ' - ' + grape_by_country['Grape'], grape_by_country['number_of_bottle'])
    plt.xticks(rotation=90)
    plt.xlabel('Country - Grape')
    plt.ylabel('Count')
    plt.title('Count of Each Grape by Country')
    plt.tick_params(axis='both', which='major', rotation=90, labelsize=6)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)
    st.dataframe(grape_by_country)


    st.write('4. Number of Bottles per Type by Region')
    bottles_per_type = wine_dataset_clean.groupby(['Type', 'Region']).size().reset_index(name='Count')
    # Pivot the data to get 'Type' as columns, 'Region' as rows, and 'Count' as values
    pivot_table = bottles_per_type.pivot(index='Type', columns='Region', values='Count').fillna(0)
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_table.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Number of Bottles per Type by Region')
    ax.set_xlabel('Wine Type')
    ax.set_ylabel('Number of Bottles')
    ax.set_xticks(range(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.index, rotation=45)
    ax.legend(title='Region')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    st.dataframe(pivot_table)

    
    st.write('5. Wine Price Distribution by Country')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(x='Country', y='Price', data=wine_dataset_clean, palette="Set3", ax=ax)
    ax.set_title('Wine Price Distribution by Country')
    ax.set_xlabel('Country')
    ax.set_ylabel('Price (£)')
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)


    st.write("6. Average Wine Price by Country")
    avg_price_per_country = df.groupby('Country')['Price'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=avg_price_per_country, y=avg_price_per_country.index, palette="viridis", ax=ax)
    ax.set_title('Average Wine Price by Country')
    ax.set_xlabel('Average Price (£)')
    ax.set_ylabel('Country')
    st.pyplot(fig)
    st.dataframe(avg_price_per_country)
        
    
    st.write('7. Distribution of Wines by Country')
    country_distribution = df['Country'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.pie(country_distribution, labels=country_distribution.index, autopct='%1.1f%%', 
        startangle=140, colors=sns.color_palette('viridis', len(country_distribution)))
    ax.set_title('Distribution of Wines by Country')
    ax.axis('equal')
    st.pyplot(fig)
    st.dataframe(country_distribution)
    
    
    st.write('8. Correlation Heatmap of Wine Features')
    # Calculate the correlation matrix
    correlation_matrix = df[['Price', 'ABV', 'Vintage']].corr()    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, )
    ax.set_title('Correlation Heatmap of Wine Features')
    st.pyplot(fig)
    
    
    st.write('9. Average Price vs Vintage Year and ABV')
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['Vintage'], df['Price'], c=df['ABV'], cmap='viridis', s=100)
    ax.set_xlabel('Vintage Year')
    ax.set_ylabel('Price')
    ax.set_title('Price vs Vintage Year and ABV')
    fig.colorbar(scatter, label='ABV')
    st.pyplot(fig)
    st.write('Grouped Data (Vintage & ABV):')
    
# ---Modeling---
if selected == Models:
    st.header('Research about :blue[Wine Produce]', divider='grey')
    st.subheader(Models)
        
    X = df[['Vintage', 'ABV']]
    y = df['Price']

    #split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #fit the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    #predict and pvaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    st.title("Wine Price Prediction Model")
    st.write("### Model: Predicting Wine Price based on Vintage and ABV")

    #show data
    st.write("#### Data:")
    st.write(df)

    #show MSE result
    st.write(f"Mean Squared Error for the model: {mse:.2f}")

    #visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_test['Vintage'], y_test, color='blue', label='Actual')
    ax.scatter(X_test['Vintage'], y_pred, color='red', label='Predicted')
    ax.set_title('Price vs Vintage and ABV')
    ax.set_xlabel('Vintage')
    ax.set_ylabel('Price')
    ax.legend()

    st.pyplot(fig)

    #show the predict result
    st.write("#### Predicted vs Actual Prices:")
    comparison_df = pd.DataFrame({
        'Vintage': X_test['Vintage'],
        'Actual Price': y_test,
        'Predicted Price': y_pred
    })
    st.write(comparison_df)