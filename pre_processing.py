def r_to_python_filter_final(r_filter_string, dataframe_name='train_df'):
    """
    Converts an advanced R-style filter string to a Python (pandas) filter string with separate conditions.

    :param r_filter_string: The advanced filter string used in R.
    :param dataframe_name: The name of the dataframe in the Python script.
    :return: A Python (pandas) compatible filter string.
    """

    # Function to convert match object to pandas format
    def convert_match_to_pandas(match):
        column_index = int(match.group(1)) - 1
        return f"{dataframe_name}.iloc[:, {column_index}]"

    # Extract conditions within c(...)
    conditions = re.findall(r"%in% c\('(.*?)'\)", r_filter_string)

    # Process each set of conditions
    all_python_conditions = []
    for condition in conditions:
        # Split individual conditions
        individual_conditions = condition.split(' & ')

        # Process each individual condition
        for ind_cond in individual_conditions:
            # Replace R column selection with Python's .iloc and adjust for zero-based indexing
            python_condition = re.sub(r'X\[,(\d+)\]', convert_match_to_pandas, ind_cond)

            # Add the processed condition to the list
            all_python_conditions.append('(' + python_condition + ')')

    # Join all conditions with '&'
    python_filter = ' & '.join(all_python_conditions)

    return python_filter
######################################################################################
def create_rule_df(R_rules_df,column,train_df):
    import pandas as pd
    #Convert the imported dataframe of rules into a list
    lt = R_rules_df[column].tolist()
    iloc_rules = []
#Convert each rule in R to python and store in a list
    for l in lt:
        rl = r_to_python_filter_final(l)
        iloc_rules.append(rl)

    cl_lt = []
    un_cl_ct = []
    row_count = []
    #Implement each rule on the training dataframe to get the respective outcome and corresponding row count 
    for l in iloc_rules: 
        x= train_df[eval(l)]
        unique_values = x['Cluster'].unique()
        total_rows = x.shape[0]
        unique_values_set = set(map(str, unique_values))
        unique_values_count = len(unique_values_set)
        row_count.append(total_rows)
        cl_lt.append(unique_values_set)
        un_cl_ct.append(unique_values_count)
    #Create a dataframe with the rules and their outcomes
    final_rules_df = pd.DataFrame({'rules': iloc_rules, 'cluster_count':un_cl_ct , 'unique_clusters_tot':cl_lt,'row_count':row_count })
    #Consider only rules with unique outcomes as a few rules truncated by the inTrees algorithm yield multiple outcomes however with low row counts
    final_rules_df = final_rules_df[final_rules_df['cluster_count']==1]
    final_rules_df['unique_clusters'] = final_rules_df['unique_clusters_tot'].apply(lambda x: next(iter(x)) if x else None)
    #Store unique outcomes as strings
    final_rules_df = final_rules_df.drop(columns=['unique_clusters_tot'])
    return final_rules_df
#####################################################################################################################
def initial_filtering(final_rules, cluster_interest,train_df):  
    filtered_xdf = train_df[train_df['Cluster'] == cluster_interest]

    # The index of filtered_df now holds the original row numbers
    original_row_numbers = filtered_xdf.index.tolist()


    filtered_rules = final_rules[final_rules['unique_clusters'] == cluster_interest]
    # Sort the filtered_rules DataFrame in descending order of 'row_count'
    filtered_rules = filtered_rules.sort_values(by='row_count', ascending=False)

    # Initialize an empty list to store the row numbers for each rule
    filtered_row_numbers = []

    # Iterate through each rule in the sorted filtered_rules DataFrame
    for rule in filtered_rules['rules']:
        # Dynamically apply the filter to x_df
        filtered_df = train_df[eval(rule)]

        # Get the row numbers and append them to the list
        row_numbers = filtered_df.index.tolist()
        filtered_row_numbers.append(row_numbers)
        # filtered_row_numbers now contains the row numbers for each rule
    return filtered_rules,original_row_numbers, filtered_row_numbers

#####################################################################################################################
#Ascertain the unique rows covered by each rule
def calculate_coverage_and_rows(selected_rules,x_df,original_row_numbers):
    covered_row_numbers = set()
    proportion_lt = []
    for condition in selected_rules:
        filtered_x_df = x_df[eval(condition)]
        covered_row_numbers.update(filtered_x_df.index.tolist())    
    
    proportion_common = len(covered_row_numbers.intersection(set(original_row_numbers))) / len(original_row_numbers) if original_row_numbers else 0
        # Calculate the proportion of original rows covered by the selected conditions
    return proportion_common, covered_row_numbers
####################################################################################################################################
#iteratively select individual rules for creating the subset representing 99%
def select_next_rule(filtered_rules, filtered_row_numbers, covered_row_numbers, original_row_numbers):
    
    uncovered_row_numbers = list(set(original_row_numbers)-covered_row_numbers)
    prop_lt = []

    for c in range(1,10000):
        combined_set = set().union(*filtered_row_numbers[c-1:c])

        # Find the common indices between the combined set and the individual list
        common_indices = combined_set.intersection(uncovered_row_numbers)

        # Calculate the proportion of common indices
        proportion_common = len(common_indices) / len(original_row_numbers) if original_row_numbers else 0
        prop_lt.append(proportion_common)

    sorted_list = sorted(enumerate(prop_lt), key=lambda x: x[1], reverse=True)

    # Getting the top element
    top_rule_index = sorted_list[0][0]
    next_rule = filtered_rules.iloc[top_rule_index,1]
    return next_rule,top_rule_index
#########################################################################################################################################

def prune_redundant_conditions(dataframe, rule):
    conditions = rule.split(' & ')
    pruned_conditions = conditions.copy()

    for condition in conditions:
        # Create an altered rule by removing the current condition
        altered_conditions = [cond for cond in pruned_conditions if cond != condition]
        altered_rule = ' & '.join(altered_conditions)

        # Convert conditions to boolean masks and apply them
        original_mask = eval(' & '.join([cond for cond in pruned_conditions]))
        altered_mask = eval(' & '.join([ cond for cond in altered_conditions]))

        # Filter dataframe using the masks
        original_filtered_df = dataframe[original_mask]
        altered_filtered_df = dataframe[altered_mask]

        # Check if the filtered dataframes are the same
        if original_filtered_df.equals(altered_filtered_df):
            # If the same, remove the redundant condition
            pruned_conditions.remove(condition)

    # Reconstruct the pruned rule
    pruned_rule = ' & '.join(pruned_conditions)
    return pruned_rule
########################################################################################################################################
def iteratively_find_most_effective_rule(dataframe, rule, cluster_of_interest):
    previous_rule = rule
    while True:
        conditions = previous_rule.split(' & ')
        effective_rules = []

        for condition in conditions:
            # Create an altered rule by removing the current condition
            altered_conditions = [cond for cond in conditions if cond != condition]
            altered_rule_str = ' & '.join(altered_conditions)

            # Apply the altered rule using eval to create a boolean mask
            altered_mask = eval(altered_rule_str)
            filtered_df = dataframe[altered_mask]

            # Check the criteria
            cluster_count = filtered_df[filtered_df['Cluster'] == cluster_of_interest].shape[0]
            total_count = filtered_df.shape[0]
            rest_count = total_count - cluster_count
            proportion = cluster_count / total_count if total_count > 0 else 0

            if cluster_count > 0 and proportion > 0.99 and rest_count < 40:
                effective_rules.append((altered_rule_str, cluster_count))

        # Find the rule with the highest count of the cluster of interest
        most_effective_rule = max(effective_rules, key=lambda x: x[1])[0] if effective_rules else None

        # Break the loop if the most effective rule is the same as the previous rule
        if most_effective_rule == previous_rule or most_effective_rule is None:
            break

        previous_rule = most_effective_rule

    return previous_rule
###############################################################################################################
def rule_selection_initial(train_df,R_rules_df,R_column, label_column, coverage_threshold = 0.95):
    rules_df = create_rule_df(R_rules_df,R_column,train_df)
    rule_dict = {}
    unique_outcomes = train_df[label_column].unique().tolist()
    for cluster_interest in unique_outcomes:
        filtered_rules,original_row_numbers, filtered_row_numbers = initial_filtering(rules_df, cluster_interest,train_df)
        selected_rules = []
        covered_row_numbers = []
        dummy_dict = {}
        # Enter the iterative process
        while coverage_proportion < coverage_threshold:
            # Select the next rule to potentially add to the selected set
            next_rule,top_rule_index = select_next_rule(filtered_rules, filtered_row_numbers, covered_row_numbers, original_row_numbers)
            next_rule = str(next_rule) #ensuring the variable type is maintained
            # If no next rule found, break the loop
            if not next_rule:
                break
            # Prune the next rule for redundancy and refine for effectiveness
            pruned_rule = prune_redundant_conditions(train_df, next_rule)
            pruned_and_refined_rule = iteratively_find_most_effective_rule(train_df, pruned_rule, cluster_interest)
            # Update the selected rules with the pruned and refined next rule
            selected_rules.append(pruned_and_refined_rule)
            # Recalculate the union of row numbers and the coverage proportion
            coverage_proportion, covered_row_numbers = calculate_coverage_and_rows(selected_rules,train_df,original_row_numbers)
        dummy_dict['rules'] = selected_rules
        dummy_dict['coverage'] = coverage_proportion
        rule_dict[cluster_interest]= dummy_dict
        # return the selected rules for the outcomes and thier respective coverage proportion
    return rule_dict
###############################################################################################################################################

def split_rule_into_conditions(rule):
    """
    Split the rule string into individual conditions.
    Assumes conditions are combined using '&' for simplicity.
    """
    conditions = [cond.strip() for cond in rule.split('&')]
    return conditions
##########################################################################################################################################################
def parse_condition(condition):
    
    """
    Extract the column index, operator, and limit from a condition string.
    """
    match = re.search(r'\(train_df\.iloc\[:, (\d+)\](<=|>=|<|>)(-?\d+\.?\d*)\)', condition)
    if match:
        column_index = int(match.group(1))
        operator = match.group(2)
        limit = float(match.group(3))
        return column_index, operator, limit
    else:
        return None, None, None
##########################################################################################################################################################
def create_altered_condition(column_index, operator, altered_limit):
    """
    Construct an altered condition string from its components.
    """
    return f"(train_df.iloc[:, {column_index}]{operator}{altered_limit})"
##########################################################################################################################################################
def update_rule_with_altered_condition(altered_rule, altered_condition, original_condition):
    """
    Insert or replace a condition in the rule list with an altered condition.
    """
    if original_condition in altered_rule:
        index = altered_rule.index(original_condition)
        altered_rule[index] = altered_condition
    else:
        altered_rule.append(altered_condition)
    return altered_rule
########################################################################################################################################################
def evaluate_rule(x_df, rule, cluster_col, cluster_value):
    """
    Check if the rule meets the acceptance criteria when applied to x_df.
    """
    # Dynamically evaluate the rule on the DataFrame
    filtered_df = x_df[eval(rule)]
    cluster_count = filtered_df[cluster_col].eq(cluster_value).sum()
    non_cluster_count = len(filtered_df) - cluster_count
    
    # Check if only one cluster exists in this filtered df
    if cluster_count / len(filtered_df) ==1 and non_cluster_count ==0:
        return True
    else:
        return False
##############################################################################################################################################
def iterate_decimal_digits(original_limit):
    """
    Generator to iterate through decimal digits of the limit.
    Yields successive decimal digits to add to the integer part.
    """
    original_str = str(original_limit)
    if '.' in original_str:
        decimal_part = original_str.split('.')[1]
        for i in range(1, len(decimal_part) + 1):
            yield decimal_part[:i]
#################################################################################################################################################
def apply_decimal_iteration(original_limit, digit):
    """
    Apply iteration over decimal digits to the limit.
    Combines the integer part with the provided decimal part.
    """
    int_part = int(np.floor(float(original_limit)))  # Ensure we always get the integer part correctly
    return float(f"{int_part}.{digit}")
#############################################################################################################################################
def alter_rule_condition(rule, x_df, cluster_col, cluster_value):
    """
    Alter each condition in the rule according to the guidelines, ensuring only the first successful
    alteration for each condition is kept.
    
    Parameters:
    - rule: The original rule as a string.
    - x_df: The DataFrame to apply the rules.
    - cluster_col: Column name for the cluster.
    - cluster_value: The value of the cluster of interest.
    
    Returns:
    - The altered rule that meets the acceptance criteria.
    """
    conditions = split_rule_into_conditions(rule)
    final_conditions = []  # To store the final set of conditions
    
    for condition in conditions:
        altered_condition = condition  # Start with the original condition as default
        column_index, operator, original_limit = parse_condition(condition)
        
        # Attempt the first alteration based on operator type
        if operator in ['<', '<=']:
            altered_limit = np.ceil(float(original_limit))
        elif operator in ['>', '>=']:
            altered_limit = np.floor(float(original_limit))
        else:
            continue  # Skip if operator is not recognized
        
        # Create a test rule with the altered condition
        test_conditions = final_conditions + [create_altered_condition(column_index, operator, altered_limit)] + conditions[conditions.index(condition)+1:]
        test_rule = ' & '.join(test_conditions)
        
        # Evaluate the test rule
        if evaluate_rule(x_df, test_rule, cluster_col, cluster_value):
            # If successful, update the altered condition
            altered_condition = create_altered_condition(column_index, operator, altered_limit)
        else:
            # If the first alteration fails, attempt iterative alterations
            for digit in iterate_decimal_digits(original_limit):
                altered_limit = apply_decimal_iteration(original_limit, digit)
                test_conditions = final_conditions + [create_altered_condition(column_index, operator, altered_limit)] + conditions[conditions.index(condition)+1:]
                test_rule = ' & '.join(test_conditions)
                
                if evaluate_rule(x_df, test_rule, cluster_col, cluster_value):
                    altered_condition = create_altered_condition(column_index, operator, altered_limit)
                    break
        
        # Append the successful alteration (or the original condition if no alteration was successful)
        final_conditions.append(altered_condition)
    
    return ' & '.join(final_conditions)
####################################################################################################################################################################################
def rule_selection(train_df,R_rules_df,R_column, label_column, coverage_threshold = 0.95):
    final_rules_dict = {}
    rules_dict = rule_selection_initial(train_df,R_rules_df,R_column, label_column, coverage_threshold)
    for cluster_value in rules_dict.keys():
        lt = []
        for rule in rules_dict[cluster_value]['rules']:
            altered_rule = alter_rule_condition(rule, train_df, label_column, cluster_value)
            lt.append(altered_rule)
        final_rules_dict[cluster_value] = lt
    
    return final_rules_dict
