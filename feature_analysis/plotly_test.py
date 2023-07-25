
# This code should be able to 1) toggle visibility for subpopulations, 2) change the


import pandas as pd
import plotly.graph_objects as go

embedding_df = pd.read_csv('/mnt/DATA1/anton/pipeline_files/feature_analysis/embeddings/testin_10LD_vanilla-VAE_1epochs.csv')
feature_df = pd.read_csv('/mnt/DATA1/anton/pipeline_files/feature_analysis/features/lowres_dataset_selection_features.csv')
df = pd.concat([feature_df, embedding_df], axis=1)
print(df.head())

fig = go.Figure()

def update_labels(attr):
    fig.data = []  # Clear existing traces
    fig.layout = {}

    for i, value in enumerate(df[attr].unique()):
        filtered_df = df[df[attr] == value]
        # fig.add_trace(go.Scatter(x=filtered_df['Dimension 0'], y=filtered_df['Dimension 1'], mode='markers', text=filtered_df[attr], name=str(value)))
        fig.data[i] = go.Scatter(x=filtered_df['Dimension 0'], y=filtered_df['Dimension 1'], mode='markers', text=filtered_df[attr], name=str(value))


    # Create buttons for each feature in the DataFrame
    buttons = []
    for feature in df.columns:
        if feature not in ['Dimension 0', 'Dimension 1']:
            buttons.append({
                'label': feature,
                'method': 'update',
                'args': [{'visible': [True] * len(df) if feature == attr else [False]*len(df) for attr in df.columns}]
            })

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                showactive=True,
                x=0.1,
                y=1.15
            ),
        ]
    )


# fig.add_trace(go.Scatter(x=df['Dimension 0'], y=df['Dimension 1'], mode='markers', text=df['day'], name='day'))  # Set initial trace
update_labels('strain')
fig.data[0].on_click(update_labels)

fig.show()
