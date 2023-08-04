def identify_features(df):
    """
    Parameters:
    df (dataframe): the dataframe to analyze

    Returns:
    dict: a dictionary of constant, numerical, and categorical features 
    """
    # Identify constant, numerical, and categorical features
    constant_features = [col for col in df.columns if df[col].nunique() <= 1]
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return {
        "constant_features": constant_features,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features
    }
def condensed_format(num, decimals=1):
    """
    Convert larger numbers into a condensed, readable format.

    Parameters:
    num (int or float): The number to convert.
    decimals (int): The number of decimal places in the output string.

    Returns:
    str: The condensed, human-readable number.
    """
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{:.{}f}{}".format(num, decimals, ['', 'k', 'M', 'B', 'T'][magnitude])
  
def check_all_types(df):
    """
    Analyzes the types of values present in each column of the given DataFrame.

    Parameters:
    df (dataframe): The dataframe to analyze.

    Returns:
    dataframe: A dataframe that summarizes the types of values and their counts and proportions in each column.
    """
    type_columns = {}
    for column in df.columns:
        value_types = df[column].map(lambda x: type(x).__name__).value_counts().to_dict()
        total_values = sum(value_types.values())
        for k, v in value_types.items():
            type_columns[column] = type_columns.get(column, {})
            type_columns[column][f"num_{k}"] = v
            type_columns[column][f"proportion_{k}"] = v / total_values

    type_df = pd.DataFrame.from_dict(type_columns, orient='index').fillna(0)
    type_df.reset_index(inplace=True)
    type_df = type_df.rename(columns={"index": "feature"})
    return type_df


def melt_dataframe(df):
    """
    Transforms a dataframe into a melted format to facilitate visualization. This focuses on proportion columns.

    Parameters:
    df (dataframe): The dataframe to melt.

    Returns:
    dataframe: A melted dataframe with columns for feature, datatype, and percentage of rows.
    """
    id_vars = ['feature']
    value_vars = [col for col in df.columns if col != 'feature' and 'proportion' in col]
    melted_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='datatype', value_name='% of Rows')
    melted_df['datatype'] = melted_df['datatype'].apply(lambda x: x.split('_')[-1])  # Remove "proportion_" prefix
    melted_df = melted_df[melted_df['% of Rows'] != 0]
    return melted_df

def plot_type_counts(melted_df, df):
    """
    Creates and shows a horizontal bar plot that visualizes datatype consistency for each feature in the original dataframe.

    Parameters:
    melted_df (dataframe): The melted dataframe with datatype information.
    df (dataframe): The original dataframe, used to calculate actual counts.
    """
    total_rows = len(df)
    melted_df['% of Rows'] = melted_df['% of Rows']
    melted_df['text'] = melted_df.apply(lambda row: f"{row['% of Rows']:.2f}% ({condensed_format(int(row['% of Rows'] * total_rows / 100))})" if 'proportion' in row['datatype'] else f"{condensed_format(int(row['% of Rows']))}", axis=1)

    fig = px.bar(melted_df, x='% of Rows', y='feature', color='datatype', orientation='h',
                 text='text',
                 color_discrete_sequence=px.colors.qualitative.Set3,
                 labels={'% of Rows': '% of Rows', 'feature': 'Feature'},
                 title='Datatype Consistency by Feature',
                 height=800)  # Set the chart height

    fig.update_traces(texttemplate='%{text}', textposition='outside', textfont_size=10)
    fig.update_layout(showlegend=True,
                      xaxis=dict(tickangle=45, tickformat=".2%"),  # Add tickformat to display as percentages
                      legend_title='Datatype',
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      font=dict(family="Times New Roman"))  # Set the font to Times New Roman

    fig.show()


def datatypes(df, return_info=False):
    """
    Performs a complete analysis of datatypes in the given DataFrame and plots the results. Optionally, it can return the detailed analysis.

    Parameters:
    df (dataframe): The dataframe to analyze.
    return_info (bool, optional): If True, returns a dataframe summarizing the analysis. Defaults to False.

    Returns:
    dataframe (optional): A dataframe that summarizes the types of values and their counts and proportions in each column. Returned only if return_info is True.
    """
    type_df = check_all_types(df)
    melted_df = melt_dataframe(type_df)
    plot_type_counts(melted_df, df)

    if return_info:
        return type_df

def compute_duplicates(df, unique_identifier_column):
    """
    Compute the number and percentage of duplicate entries in a given column of a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    unique_identifier_column (str): The column name of the unique identifier.

    Returns:
    pandas.DataFrame: A DataFrame with the number and percentage of duplicate entries in the specified column.
    """
    value_counts = df[unique_identifier_column].value_counts()
    count_frequency = value_counts.value_counts()
    total = float(value_counts.sum())
    count_frequency_percent = (count_frequency / total)

    count_frequency.index = count_frequency.index - 1
    count_frequency_percent.index = count_frequency_percent.index - 1

    count_frequency_df = pd.DataFrame({
        'num_unique_identifiers': count_frequency,
        'proportion_unique_identifiers': count_frequency_percent
    })
    count_frequency_df = count_frequency_df.sort_index(ascending=True)
    count_frequency_df.reset_index(inplace=True)
    count_frequency_df.rename(columns={'index': 'num_duplicates'}, inplace=True)
    return count_frequency_df

def plot_duplicates(duplicates_df):
    """
    Generate a bar plot of the percentage of duplicate entries in a DataFrame

    Parameters:
    duplicates_df (pandas.DataFrame): The DataFrame to plot, as returned by compute_duplicates(df).
    """
    text_labels = [f"{row['proportion_unique_identifiers']*100:.1f}% ({condensed_format(row['num_unique_identifiers'])})"
                   for index, row in duplicates_df.iterrows()]

    fig = px.bar(duplicates_df, x='num_duplicates', y='proportion_unique_identifiers',
                 text=text_labels,
                 labels={'num_duplicates': '# of Duplicates', 'proportion_unique_identifiers': '% of Unique Identifiers'},
                 title="% of 'Unique' Identifiers by # of Duplicates",
                 height=600)

    fig.update_traces(texttemplate='%{text}', textposition='outside', textfont_size=10, marker_color='rgba(10, 10, 70, 0.7)')
    fig.update_layout(showlegend=False,
                      xaxis=dict(tickmode='array', tickvals=np.arange(0, duplicates_df['num_duplicates'].max() + 1), tickangle=45),
                      yaxis=dict(tickformat=".0%"),
                      font=dict(family="Times New Roman"))  # Set the font to Times New Roman

    return fig.show()



def duplicates(df, unique_identifier_column, return_info=False):
    """
    Compute and plot the number and percentage of duplicate entries for each unique identifier in a dataframe 

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    unique_identifier_column (str): The column name of the unique identifier.
    return_info (bool): If True, return the DataFrame of duplicate entries information.

    Returns:
    pandas.DataFrame or None: If return_info is True, return a DataFrame with duplicate entries information.
    """
    duplicates_df = compute_duplicates(df, unique_identifier_column)
    plot_duplicates(duplicates_df)
    if return_info:
        return duplicates_df



def compute_missing(df):
    """
    Compute the number and proportion of missing values for each feature in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
    pandas.DataFrame: A DataFrame containing the number and proportion of missing values for each feature.
    """
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    missing_percentage = (missing_values / len(df))

    missing_df = pd.DataFrame({
        'num_rows_w_missing_data': missing_values,
        'proportion_rows_w_missing_data': missing_percentage
    })
    missing_df = missing_df.sort_values(by='proportion_rows_w_missing_data', ascending=False)
    missing_df.reset_index(inplace=True)
    missing_df.rename(columns={'index': 'feature'}, inplace=True)
    return missing_df

def plot_missing(missing_df, color='rgba(10, 10, 70, 0.7)'):
    """
    Generate a bar plot of the percentage of missing values by feature in a DataFrame.

    Parameters:
    missing_df (pandas.DataFrame): The DataFrame to plot, as returned by compute_missing(df).
    color (str, optional): The color of the bars in the plot. Defaults to 'rgba(10, 10, 70, 0.7)'.

    Returns:
    None: Displays the plot.
    """
    fig = px.bar(missing_df, x='feature', y='proportion_rows_w_missing_data',
                 text=[f"{row['proportion_rows_w_missing_data']*100:.1f}% ({condensed_format(row['num_rows_w_missing_data'])})" for index, row in missing_df.iterrows()],
                 labels={'feature': 'Feature', 'proportion_rows_w_missing_data': '% of Rows with Missing Data'},
                 title='% of Rows with Missing Data by Feature',
                 height=600)

    fig.update_traces(texttemplate='%{text}', textposition='outside', textfont_size=10, marker_color=color)
    fig.update_layout(showlegend=False,
                      xaxis=dict(tickangle=45),
                      yaxis=dict(tickformat=".0%"),
                      font=dict(family="Times New Roman"))  # Set the font to Times New Roman

    fig.show()


def missing(df, color='rgba(10, 10, 70, 0.7)', return_info=False):
    """
    Compute and plot the number and percentage of missing values for each feature in a dataframe. Optionally, return the detailed analysis.

    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    color (str, optional): The color of the bars in the plot. Defaults to 'rgba(10, 10, 70, 0.7)'.
    return_info (bool): If True, return the DataFrame of missing entries information. Defaults to False.

    Returns:
    pandas.DataFrame or None: If return_info is True, return a DataFrame with missing entries information. Otherwise, return None.
    """
    missing_df = compute_missing(df)
    plot_missing(missing_df, color)
    if return_info:
        return missing_df

def calculate_z_scores(df, numerical_features):
    """
    Calculate the Z scores for the provided numerical features in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the numerical features.
    numerical_features (list of str): List of column names corresponding to numerical features.

    Returns:
    numpy.ndarray: An array containing the Z scores for the specified numerical features.
    """
    return scipy.stats.zscore(df[numerical_features])

def get_outlier_positions(z_scores, z_threshold):
    """
    Get the positions of outliers in the given Z scores based on the provided threshold.

    Parameters:
    z_scores (numpy.ndarray): The Z scores of the values.
    z_threshold (float): The Z score threshold to use to determine outliers.

    Returns:
    tuple: Two boolean arrays indicating the positions of left and right outliers.
    """
    return (z_scores < -z_threshold), (z_scores > z_threshold)

def outliers_by_zscore(df, numerical_features, z_threshold):
    """
    Calculate the outliers using the Z scores for the specified numerical features in a DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    numerical_features (list of str): List of column names corresponding to numerical features.
    z_threshold (float): The Z score threshold to use to determine outliers.

    Returns:
    pandas.DataFrame: A DataFrame containing the number and proportion of left and right outliers for each feature.
    """
    z_scores = calculate_z_scores(df, numerical_features)
    left_outliers_positions, right_outliers_positions = get_outlier_positions(z_scores, z_threshold)

    left_outliers = df[left_outliers_positions]
    right_outliers = df[right_outliers_positions]

    left_outliers_counts = left_outliers.count()
    right_outliers_counts = right_outliers.count()

    # Calculate the percentage of outliers
    total_counts = len(df)
    left_outliers_proportion = (left_outliers_counts / total_counts)
    right_outliers_proportion = (right_outliers_counts / total_counts)

    outlier_counts = pd.DataFrame({
        'num_left_outliers': left_outliers_counts,
        'proportion_left_outliers': left_outliers_proportion,
        'num_right_outliers': right_outliers_counts,
        'proportion_right_outliers': right_outliers_proportion
    })

    # Reset index and rename the new column to 'Features'
    outlier_counts = outlier_counts.reset_index().rename(columns={"index": "feature"})

    return outlier_counts


def autolabel(ax, rects, counts, percentages, precision=2):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to annotate.
    rects (list): The bar objects to annotate.
    counts (list): The counts corresponding to each bar.
    percentages (list): The percentages corresponding to each bar.
    precision (int, optional): The number of decimal places to show in the annotation. Defaults to 2.

    Returns:
    None: Adds annotations directly to the plot.
    """
    for rect, count, percentage in zip(rects, counts, percentages):
        height = rect.get_height()

        # Only annotate when the count is not zero
        if count != 0:
            height_str = '{:.{}f}%'.format(percentage * 100, precision)
            count_str = '{:,}'.format(count)

            ax.annotate('{} ({})'.format(height_str, count_str),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15)  # Reduced font size



def plot_outliers(df, numerical_features, z_threshold=7.13, precision=2):
    """
    Plot bar plots for each numerical feature in the DataFrame, showing the percentage of outliers.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the data.
    numerical_features (list of str): List of numerical features to consider.
    z_threshold (float, optional): The Z score threshold to use to determine outliers. Defaults to 7.13.
    precision (int, optional): Number of decimal places to show in y-axis labels. Defaults to 2.

    Returns:
    None: Displays the plot.
    """
    z_scores = calculate_z_scores(df, numerical_features)
    left_outliers_positions, right_outliers_positions = get_outlier_positions(z_scores, z_threshold)

    left_outliers_counts = np.count_nonzero(left_outliers_positions, axis=0)
    right_outliers_counts = np.count_nonzero(right_outliers_positions, axis=0)

    # Calculate the percentage of outliers
    total_counts = len(df)
    left_outliers_proportions = (left_outliers_counts / total_counts)
    right_outliers_proportions = (right_outliers_counts / total_counts)

    # Creating a DataFrame to be used with Plotly
    plot_data = pd.DataFrame({
        'Feature': np.tile(numerical_features, 2),
        'Percentage': np.concatenate([left_outliers_proportions, right_outliers_proportions]),
        'Count': np.concatenate([left_outliers_counts, right_outliers_counts]),
        'Side': ['Left Outliers'] * len(numerical_features) + ['Right Outliers'] * len(numerical_features)
    })

    text_labels = [f"{row['Percentage']*100:.1f}% ({condensed_format(row['Count'])})" for index, row in plot_data.iterrows()]

    # Define custom color scheme
    color_discrete_map = {
        'Left Outliers': 'red',
        'Right Outliers': 'green'
    }

    fig = px.bar(plot_data, x='Feature', y='Percentage', color='Side',
                 title='% of Rows with Outliers by Feature',
                 barmode='group',
                 color_discrete_map=color_discrete_map)  # Add the custom color scheme here

    y_offset = 0.02  # You can adjust this value to position the annotations as you like
    x_offset = 0.15  # You can adjust this value to position the annotations as you like

    # Create custom annotations for non-zero counts
    for index, row in plot_data.iterrows():
        if row['Count'] != 0:  # Only add annotation if count is non-zero
            text_label = f"{row['Percentage']*100:.1f}% ({condensed_format(row['Count'])})"
            # Determine the x position based on the index of the feature within numerical_features
            x_position = numerical_features.index(row['Feature'])
            # Apply the offset based on whether it's a left or right outlier
            if row['Side'] == 'Right Outliers':
                x_position += x_offset  # Adjust the x position for right outliers
            elif row['Side'] == 'Left Outliers':
                x_position -= x_offset  # Adjust the x position for left outliers
                fig.add_annotation(
                    x=x_position,
                    y=row['Percentage'] + y_offset,
                    text=text_label,
                    showarrow=False,
                    font=dict(size=12),  # Increase the text font size here
                    textangle=-90        # Rotate text by 90 degrees to make it vertical
                )


    # Customizing the layout
    fig.update_layout(
        showlegend=True,
        yaxis_tickformat=f".{precision}%",
        xaxis_title='Feature',
        yaxis_title='Outliers as a % of Rows',
        font=dict(family="Times New Roman") # Set the font family to Times New Roman
    )

    # Show the plot
    fig.show()


