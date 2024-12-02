import matplotlib.pyplot as plt


def get_frequencies(col):

    # we want to get the unique values from this column
    # no duplicates values and unordered set but make sorted with ordered list
    unique_col_values = sorted(list(set(col)))
    counts = []
    for val in unique_col_values:
        counts.append(col.count(val))

    return unique_col_values, counts


def plot_histogram(unique_values, counts, title, xlabel, ylabel):
    plt.bar(unique_values, counts, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Function to convert MPG to DOE fuel economy rating
def mpg_to_rating(mpg):
    mpg = round(mpg)  # Round the MPG value to the nearest whole number
    if mpg >= 45:
        return 10
    elif 37 <= mpg <= 44:
        return 9
    elif 31 <= mpg <= 36:
        return 8
    elif 27 <= mpg <= 30:
        return 7
    elif 24 <= mpg <= 26:
        return 6
    elif 20 <= mpg <= 23:
        return 5
    elif 17 <= mpg <= 19:
        return 4
    elif 15 <= mpg <= 16:
        return 3
    elif mpg == 14:
        return 2
    elif mpg <= 13:
        return 1
    else:
        return 0  # In case of invalid MPG values


def convert_mpg_to_ratings(mpg_list):
    # convert each MPG value to its corresponding rating
    return [mpg_to_rating(mpg) for mpg in mpg_list]


def compute_slope_intercept(x, y):
    meanx = sum(x) / len(x)
    meany = sum(y) / len(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den
    # y = mx + b -> y - mx
    b = meany - m * meanx
    return m, b

def create_bins(mpg_values, num_bins=5):
    """Create equal-width bins and return categorized values."""
    min_mpg = min(mpg_values)
    max_mpg = max(mpg_values)

    # Calculate the width of each bin
    bin_width = (max_mpg - min_mpg) / num_bins

    # Create bins and assign values
    bins = []
    bin_labels = []

    for i in range(num_bins):
        lower_bound = min_mpg + i * bin_width
        upper_bound = lower_bound + bin_width
        # Label for the bin
        bin_labels.append(f"{int(lower_bound)}--{int(upper_bound)}")
        bins.append((lower_bound, upper_bound))

    # Categorize MPG values into bins
    categorized_values = []
    for mpg in mpg_values:
        for i, (lower, upper) in enumerate(bins):
            if lower <= mpg < upper:
                categorized_values.append(i)  # Store the bin index
                break

    return categorized_values, bin_labels


def plot_histogram_with_bins(mpg_values):
    # Get the categorized bin values and labels
    categorized_values, bin_labels = create_bins(mpg_values)

    # use get_frequencies to calculate the frequency counts of the categorized
    # values
    unique_values, counts = get_frequencies(categorized_values)

    # Prepare data for plotting
    plt.bar(bin_labels, counts, color='skyblue')
    plt.title("MPG Discretization Histogram")
    plt.xlabel("MPG Bins")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


def plot_bin_histogram(data, title, xlabel):
    plt.hist(data, bins=10, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def compute_correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    # Calculate the numerator and denominator of Pearson's r
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    correlation_coefficient = numerator / (denominator_x**0.5 * denominator_y**0.5)
    return correlation_coefficient

def plot_scatter_with_regression(x, y, x_label, y_label='MPG'):
    plt.figure()
 
    # Calculate the linear regression coefficients
    m, b = compute_slope_intercept(x,y)
    print(f"Slope (m): {m}, Intercept (b): {b}")


    # Calculate correlation coefficient
    correlation_coefficient = compute_correlation(x,y)

    # Create scatter plot
    plt.scatter(x, y, color='skyblue', edgecolor='black', label='Data points')

    # Plot regression line
    
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="red", lw=3)

    # Labeling the plot
    plt.title(f'Scatter Plot: {x_label} vs {y_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    # Display the correlation coefficient
    plt.text( .9, 0.02,f'r: {correlation_coefficient:.2f}', transform=plt.gca().transAxes)
    plt.grid()
    plt.show()

def plot_mpg_by_model_year(model_years, mpg_values):
    # Create a dictionary to hold mpg values for each model year
    mpg_by_year = {}
    
    for year, mpg in zip(model_years, mpg_values):
        if year not in mpg_by_year:
            mpg_by_year[year] = []
        mpg_by_year[year].append(mpg)
    
    # Prepare data for plotting
    years = sorted(mpg_by_year.keys())
    mpg_data = [mpg_by_year[year] for year in years]

    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(mpg_data, labels=years)

    # Customize the plot
    plt.title('MPG Distribution by Model Year')
    plt.xlabel('Model Year')
    plt.ylabel('MPG')
    plt.grid(axis='y')

    # Show the plot
    plt.tight_layout()
    plt.show()
