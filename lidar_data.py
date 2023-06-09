"""Aggregate the data to one table."""
from experiment_measurement import data_aggregation_helper
#import data_aggregation_helper

"""
table_column_config  ::= table_column, [ df_aggregated_topics ]
table_column         ::= ( topic, table_column_name, func )

topic                ::= string
table_column_name    ::= string
func                 ::= TableConfig -> Object

message_dataframe    ::= pd.Series      # All messages to one topic and robot
time                 ::= number
"""
table_column_config = [
    data_aggregation_helper.TableColumn(
        'odom',
        'odom',
        lambda conf: data_aggregation_helper.get_latest_in_interval(conf)['data']
    ),
    data_aggregation_helper.TableColumn(
        'scan',
        'scan',
        lambda conf: data_aggregation_helper.get_latest_in_interval(conf)['data']
    ),
    data_aggregation_helper.TableColumn(
        '/ground_truth',
        'poses',
        lambda conf: data_aggregation_helper.get_latest_in_interval(conf)['data']
    )
]
