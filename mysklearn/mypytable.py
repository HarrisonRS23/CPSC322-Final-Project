"""
Programmer: Harrison Sheldon
Class: CSPC 322, Fall 2024
Programming Assignment #7
11/20/24

Description: This program uses different methods of the MyTable class to create tables
and modify the tables.
"""
import copy
import csv
import math
import statistics
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the
# unit tests


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """
        Extracts a column from the table data as a list.

        Args:
            col_identifier (str or int): String for a column name or int
                for a column index.
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column.

        Notes:
            Raises ValueError on invalid col_identifier.
        """
        # Determine column index
        if isinstance(
                col_identifier,
                int):  # If integer, treat as column index
            column_index = col_identifier
        elif isinstance(col_identifier, str):  # If string, match column name
            try:
                column_index = self.column_names.index(col_identifier)
            except ValueError:
                raise ValueError(f"Invalid column name: {col_identifier}")
        else:
            raise ValueError("col_identifier must be an int or str.")

        # Extract column values
        column_list = []
        for row in self.data:
            value = row[column_index]
            if include_missing_values or value != "NA":  # Include/exclude "NA"
                column_list.append(value)

        return column_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for index, val in enumerate(row):
                try:
                    row[index] = float(val)
                except ValueError:
                    continue

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.
        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # reverse the list so that each removal doesnt affect index for next
        # removal
        row_indexes_to_drop.sort(reverse=True)
        for index in row_indexes_to_drop:
            try:
                del self.data[index]
            except IndexError:
                print("fail")

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        table = []
        # open file for reading
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            for line in reader:
                table.append(line)
        # sav header
        header = table.pop(0)
        self.column_names = header
        self.data = table
        # run convert to numeric after load
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        # open file as write and write header then data
        with open(filename, 'w', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        duplicate_indexs = []  # list to return of indexes with duplicate rows found
        key_index_list = []  # list of key indexes that are converted from the column names
        duplicate_list = []  # list holding values to check if there are duplicates

        # convert key column names from a name into an index (int) value
        for val in key_column_names:
            key_index_list.append(self.column_names.index(val))
        # go through each row in the table and keep track of the index
        for row_index, row in enumerate(self.data):
            # create a tuple using the values from the key columns
            key = tuple(row[column] for column in key_index_list)
            # check to see if the key has been seen in the list (duplicate)
            if key in duplicate_list:
                # if found duplicate add the index to list to return
                duplicate_indexs.append(row_index)
            else:
                # if not a duplicate add to list as this might be a duplicate
                # with a later value
                duplicate_list.append(key)
        # return list of indexes where duplicate found
        return duplicate_indexs

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        removal_indexes = []
        for row_index, row in enumerate(self.data):
            for val in row:
                if val == "NA":
                    # create list of indexes of rows to drop
                    removal_indexes.append(row_index)
        # drop all rows at once using function
        self.drop_rows(removal_indexes)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # first go through and find average for column
        # second iteration check for every value that is null and fill in with
        # average val

        self.convert_to_numeric()
        index = self.column_names.index(col_name)
        total = 0
        total_val = 0
        for row in self.data:
            if not row[index] == "NA":
                total += int(row[index])
                total_val += 1
        # use ceiling function to calculate average value
        average = math.ceil((total / total_val))
        for row in self.data:
            # find all missing values and set to average
            if row[index] == "NA":
                row[index] = average

    def append_row(self, new_row):
        """Appends a new row to the table data.

        Args:
            new_row(list of obj): a list representing a row to append to the table.

        Notes:
            The number of elements in new_row should match the number of columns in the table.
            Raises a ValueError if the length of new_row doesn't match the column length.
        """
        if len(new_row) != len(self.column_names):
            raise ValueError(
                "The row length does not match the table's column count.")

        self.data.append(new_row)

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """

        # convert key column names from a name into an index (int) value
        col_index_list = []
        for val in col_names:
            col_index_list.append(self.column_names.index(val))

        num_of_rows = len(col_index_list)
        stats_table = [[float('inf'), float('-inf'), 0, 0, 0]
                       for x in range(num_of_rows)]  # create stats table
        # 0 - min (infinity initial val ), 1 - max (-infinity initial value),
        # 2 - mid, 3 - avg, 4 - median

        # store values to calc media and avg later
        col_values = [[] for _ in range(num_of_rows)]
        # iterate through list and keep track of min and max value
        for row in self.data:
            for x, val in enumerate(row):
                if x in col_index_list and val != "NA":
                    val = float(val)
                    index = col_index_list.index(x)
                    stats_table[index][0] = min(
                        stats_table[index][0], val)  # min value
                    stats_table[index][1] = max(
                        stats_table[index][1], val)  # max value
                    col_values[index].append(val)

        for i in range(num_of_rows):
            min_val, max_val = stats_table[i][0], stats_table[i][1]
            stats_table[i][2] = (min_val + max_val) / \
                2  # Mid = (min + max) / 2
            # find mean if there is a value else 0
            stats_table[i][3] = sum(col_values[i]) / \
                len(col_values[i]) if col_values[i] else 0
            stats_table[i][4] = statistics.median(
                col_values[i]) if col_values[i] else 0  # find median if there is value else 0

        # build stats table by appending column names to stats_table rows
        summary_data = []
        for i, col_name in enumerate(col_names):
            row = [col_name] + stats_table[i]
            summary_data.append(row)
        # check for default values to see if they have been changed and append
        # new values if they have
        valid_summary_data = []
        for row in summary_data:
            if row[1] != float('inf') and row[2] != float('-inf'):
                valid_summary_data.append(row)
        # return stats table if there are values otherwise an empty array
        return MyPyTable(
            col_names,
            valid_summary_data if valid_summary_data else [])

    def extract_key_from_row(self, key_column_names, row):
        """Extract a key from the row passed in using the key_column_names

        Args:
            row(list of str): row values to extract key from
            key_column_names(list of str): column names to use as row keys.

        Returns:
            values: list of values found based on key
        """
        key_index_list = []
        values = []
        # convert to index list with just keys then append the key to a new
        # list and return all found values as a key
        for val in key_column_names:
            key_index_list.append(self.column_names.index(val))
        for key in key_index_list:
            values.append(row[key])
        return values

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        joined_table = []

        key_index_list = []
        for val in key_column_names:
            key_index_list.append(other_table.column_names.index(val))
        key_index_list.sort(reverse=True)

        for row1 in self.data:
            # use helper function to extract key to help comparisons
            key1 = self.extract_key_from_row(key_column_names, row1)
            for row2 in other_table.data:
                key2 = other_table.extract_key_from_row(key_column_names, row2)
                if key1 == key2:

                    # create a new row that combines row1 and row2, excluding
                    # the key columns from row2
                    new_row = list(row1)  # Start with values from row1

                    # append values from row2, excluding key columns
                    for index, val in enumerate(row2):
                        if index not in key_index_list:
                            new_row.append(row2[index])

                    # Add the combined row to the joined table
                    joined_table.append(new_row)

        new_column_names = list(self.column_names)
        for col_name in other_table.column_names:
            if col_name not in key_column_names:  # check if it's not a key column
                new_column_names.append(col_name)  # append to new_column_names

        return MyPyTable(new_column_names, joined_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        joined_table = []
        matched_rows_1 = set()  # track matched rows from self.data
        matched_rows_2 = set()  # track matched rows from other_table

        # compute key index lists for both tables
        key_index_other = [other_table.column_names.index(
            col) for col in key_column_names]

        # Perform full outer join
        for i, row1 in enumerate(self.data):
            key1 = self.extract_key_from_row(key_column_names, row1)
            match_found = False

            for j, row2 in enumerate(other_table.data):
                key2 = other_table.extract_key_from_row(key_column_names, row2)

                if key1 == key2:
                    match_found = True
                    matched_rows_1.add(i)
                    matched_rows_2.add(j)

                    # create a new row that combines row1 and row2, excluding
                    # key columns from row2
                    new_row = list(row1)
                    new_row += [val for idx,
                                val in enumerate(row2) if idx not in key_index_other]
                    joined_table.append(new_row)

            # handle non-matching row1 (rows from self that have no match)
            if not match_found:
                new_row = list(row1)  # start with row1's values
                # Add NAs based on how long columns are
                new_row.extend(["NA"] *
                               (len(other_table.column_names) -
                                len(key_column_names)))
                joined_table.append(new_row)

        # add unmatched rows from other_table
        for j, row2 in enumerate(other_table.data):
            if j not in matched_rows_2:
                # start with NAs for self columns
                new_row = ["NA"] * len(self.column_names)
                for i, val in enumerate(row2):
                    if i in key_index_other:
                        new_row[self.column_names.index(
                            other_table.column_names[i])] = val
                    else:
                        new_row.append(val)
                joined_table.append(new_row)

        # Create new column names
        new_column_names = list(self.column_names)
        for col_name in other_table.column_names:
            if col_name not in key_column_names:
                new_column_names.append(col_name)

        return MyPyTable(new_column_names, joined_table)

    def csv_to_mypytable(self, file_path):
        """
        Reads a CSV file and converts it to a MyPyTable object.

        Parameters:
            file_path (str): The path to the CSV file.

        Returns:
            MyPyTable: A MyPyTable object with column names and data populated from the CSV.
        """
        with open(file_path, 'r', newline='') as file:
            reader = csv.reader(file)

            # Read the header
            column_names = next(reader)

            # Read the data rows
            data = [row for row in reader]

        # Create a MyPyTable with the read column names and data
        return MyPyTable(column_names=column_names, data=data)
    
    def add_column(self, column_name, values):
        """
        Adds a new column to the table.

        Parameters:
        column_name (str): The name of the new column to add.
        values (list): A list of values to populate the new column. Must match the number of rows in the table.

        Returns:
        None

        Raises:
        ValueError: If the number of values does not match the number of rows in the table.
        """
        if len(self.data) != len(values):
            raise ValueError("The number of values must match the number of rows in the table.")

        if column_name in self.column_names:
            raise ValueError(f"The column name '{column_name}' already exists in the table.")

        self.column_names.append(column_name)

        for row, value in zip(self.data, values):
            row.append(value)

    def remove_column(self, column_name):
        """Removes a column from the table.

        Args:
            column_name (str): The name of the column to remove.

        Raises:
            ValueError: If the column_name is not found in the table.
        """
        if column_name not in self.column_names:
            raise ValueError(f"Column '{column_name}' not found in the table.")

        # Get the index of the column to remove
        col_index = self.column_names.index(column_name)

        # Remove the column name from the column_names list
        self.column_names.pop(col_index)

        # Remove the corresponding column data from each row
        for row in self.data:
            row.pop(col_index)
