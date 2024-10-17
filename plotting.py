import pandas as pd
import matplotlib
matplotlib.use('Agg')
import plotly.io as pio
import plotly.graph_objects as go
import numpy as np

def plot_serve_stats_by_surface(data):
    """
    Plots the graph for serve performance by surface type.

    Args:
    data: Dictionary of match data.

    Returns:
    plot_html: HTML representation of graph.
    None: no graph if invalid data.
    """
    if 'matches' in data:
        df = pd.DataFrame(data['matches']['data'], columns=data['matches']['headers'])
        df.columns = [col.replace('\n', ' ') for col in df.columns]  

        df.replace('', pd.NA, inplace=True)
        
        headers = ['Surface', '1stIn', '1st%', 'A%', '2nd%']
        df = df[headers]
        df.dropna(subset=headers, inplace=True)
        
        df['1stIn'] = pd.to_numeric(df['1stIn'].astype(str).str.replace('%', '').astype(float), errors='coerce')
        df['1st%'] = pd.to_numeric(df['1st%'].astype(str).str.replace('%', '').astype(float), errors='coerce')
        df['2nd%'] = pd.to_numeric(df['2nd%'].astype(str).str.replace('%', '').astype(float), errors='coerce')

        surfaces = df['Surface'].unique()
        metrics = ['1stIn', '1st%', '2nd%']

        fig = go.Figure()

        colours = {
            'Grass': 'rgba(44, 160, 44, 0.65)',
            'Clay': 'rgba(255, 127, 14, 0.65)',
            'Hard': 'rgba(107, 174, 214, 0.65)',      
            'Carpet': 'rgba(128, 128, 128, 0.65)'    
        }

        for surface in surfaces:
            surface_data = df[df['Surface'] == surface]

            radar_data = surface_data[['1stIn', '1st%', '2nd%']].values.flatten()

            radar_data = list(radar_data) + [radar_data[0]]
            metrics_extended = metrics + [metrics[0]]
            colour = colours.get(surface)
            
            fig.add_trace(go.Scatterpolar(
                r=radar_data,
                theta=metrics_extended,
                fill='toself',
                fillcolor=colour,
                line=dict(color=colour),
                name=surface,
                text=[f"{metric}: {value}<br>Ace%: {surface_data['A%'].values[0]}" 
                    for metric, value in zip(metrics_extended, radar_data)],  
                hovertemplate='%{text}<extra></extra>',  
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, 
                    range=[0, 100],
                    linecolor='#202020',
                    gridcolor='#202020'
                ),  
                angularaxis=dict(
                    tickvals=metrics_extended,
                    visible=True,
                    linecolor='#202020',
                    gridcolor='#202020'
                )),
            showlegend=True,
            title='Serve Performance by Surface',
            template='plotly_white',
            height=720,
            width=1200,
            plot_bgcolor='#202020',
            paper_bgcolor='#202020',
            font=dict(color='#E0E0E0')
        )

        plot_html = fig.to_html(full_html=False)
        
        return plot_html
    else:
        return None

    
def plot_points_by_surface(data):
    """
    Plots box plot of Dominance Ratios by surface type.

    Args:
    data: Dictionary of match data.

    Returns:
    plot_html: HTML representation of graph.
    None: no graph if invalid data.
    """
    if 'matches' in data:

        df = pd.DataFrame(data['matches']['data'], columns=data['matches']['headers'])
        df.columns = [col.replace('\n', ' ') for col in df.columns] 
        
        headers = ['Surface', 'DR']
        df = df[headers]

        df.replace('', pd.NA, inplace=True)  

        df.dropna(subset=headers, inplace=True)

        df['Surface'] = df['Surface'].astype(str)
        df['DR'] = pd.to_numeric(df['DR'], errors='coerce')

        colours = {
            'Grass': 'rgb(44, 160, 44)',
            'Clay': 'rgb(255, 127, 14)',
            'Hard': 'rgb(107, 174, 214)',
            'Carpet': 'rgba(128, 128, 128)'
        }

        fig = go.Figure()

        for surface in df['Surface'].unique():
            surface_data = df[df['Surface'] == surface]['DR']
            fig.add_trace(go.Box(
                y=surface_data,
                name=surface,
                boxpoints='all',  
                marker_color=colours.get(surface, 'rgb(0,0,0)'),  
                line_color=colours.get(surface, 'rgb(0,0,0)'),
                jitter=0.1,
                pointpos=0,
                hovertemplate=(
                    'Court Type: %{x}<br>' 
                    'DR: %{y:.2f}<extra></extra>'  
                )
            ))
        
        fig.update_layout(
            xaxis_title='Court Type',
            yaxis_title='Dominance Ratio',
            height=720,
            width=1200,
            plot_bgcolor='#202020',
            paper_bgcolor='#202020',
            font=dict(color='#E0E0E0')
            
        )

        plot_html = fig.to_html(full_html=False)
        
        return plot_html
    else:
        return None
        
    
def plot_rank_over_time(data, category):
    """
    Plots graph of player rank over time.

    Args:
    data: Dictionary of match data.
    category: ATP or WTA categorisation.

    Returns:
    plot_html: HTML representation of graph.
    None: no graph if invalid data.
    """
    rank_column = 'ATP Rank' if category == 'LTA' else 'WTA Rank'

    if 'year-end-rankings' in data:

        rankings_df = pd.DataFrame(data['year-end-rankings']['data'], columns=data['year-end-rankings']['headers'])
        rankings_df.columns = [col.replace('\n', ' ') for col in rankings_df.columns]
        rankings_df['Year'] = rankings_df['Year'].str.extract(r'(\d{4})')
        rankings_df['Year'] = pd.to_numeric(rankings_df['Year'], errors='coerce')
        rankings_df[rank_column] = pd.to_numeric(rankings_df[rank_column], errors='coerce')
        rankings_df['Elo'] = pd.to_numeric(rankings_df['Elo'], errors='coerce') 
        rankings_df = rankings_df.dropna(subset=['Year', rank_column, 'Elo'])
    else:
        return None

    if 'tour-years' in data:
        tour_years_df = pd.DataFrame(data['tour-years']['data'], columns=data['tour-years']['headers'])
    elif 'chall-years' in data:
        tour_years_df = pd.DataFrame(data['chall-years']['data'], columns=data['chall-years']['headers'])
    else:
        return None

    tour_years_df.columns = [col.replace('\n', ' ') for col in tour_years_df.columns]
    tour_years_df['Year'] = tour_years_df['Year'].str.extract(r'(\d{4})')
    tour_years_df['Year'] = pd.to_numeric(tour_years_df['Year'], errors='coerce')
    tour_years_df['Win%'] = tour_years_df['Win%'].str.replace('%', '').astype(float)
    tour_years_df = tour_years_df.dropna(subset=['Year', 'Win%'])

    merged_df = pd.merge(rankings_df, tour_years_df, on='Year')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=merged_df['Year'],
        y=merged_df[rank_column],
        name=f'{category} Rank',
        marker=dict(color='rgba(207, 102, 121, 0.9)'),
        width=0.4,  
        yaxis='y2',
        opacity=0.5,
        hovertemplate='Year: %{x}<br>Rank: %{y}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=merged_df['Year'],
        y=merged_df['Win%'],
        mode='lines+markers',
        name='Win Percentage',
        line=dict(color='rgba(134, 139, 252, 0.9)', width=2),
        marker=dict(color='rgba(134, 139, 252, 0.9)'),
        customdata=merged_df[['Elo']].values,
        hovertemplate=(
            'Year: %{x}<br>'  
            'Win%: %{y:.2f}%<br>'  
            'Elo: %{customdata:.0f}<br>'  
            '<extra></extra>' 
        )
    ))

    max_rank = merged_df[rank_column].max()
    tickvals = []
    current = 1
    if max_rank >= 250:
        while current <= int(max_rank):
            if current == 1:
                tickvals.append(current)
                current = 100
            else:
                current += 100
            tickvals.append(current)
    else:   
        while current <= int(max_rank):
            if current == 1:
                tickvals.append(current)
                current = 10
            else:
                current += 10
            tickvals.append(current)
    ticktext = tickvals

    fig.update_layout(
        title=f'Player Performance Over Time ({category})',
        xaxis_title='Year',
        yaxis_title='Win Percentage',
        yaxis=dict(
            title='Win Percentage',
            range=[0, 100],
            gridcolor='#E0E0E0',  
            linecolor='#E0E0E0',
            zeroline=False   
        ),
        yaxis2=dict(
            title=f'{category} Rank',
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, max_rank + 10],  
            tickvals=tickvals,
            ticktext=ticktext,
            tickmode='array',
            gridcolor='#E0E0E0',
            linecolor='#E0E0E0'
        ),
        barmode='group',  
        xaxis=dict(
            tickmode='linear',
            gridcolor='#E0E0E0',  
            linecolor='#E0E0E0'
        ),
        legend=dict(
            title='Metrics',
            orientation='h',  
            yanchor='bottom',
            xanchor='right',
            y=-0.2,  
            x=1.1
        ),
        plot_bgcolor='#202020',
        paper_bgcolor='#202020',
        font=dict(color='#E0E0E0'),
        height=720,
        width=1200
    )

    plot_html = pio.to_html(fig, full_html=False)
    
    return plot_html


def plot_data(data, category, plot_type='radar'):
    """
    Plot data based on the provided category and plot type.

    Args:
        data: Player's data.
        category: Whether or not player is WTA or ATP.
        plot_type: The type of plot to create.

    Returns:
        appropriate graph: HTML representation of chosen graph.
    """
    if plot_type == 'box_plot':
        return plot_points_by_surface(data)
    elif plot_type == 'bar_graph':
        return plot_rank_over_time(data, category)
    elif plot_type == 'scatter_plot':
        return plot_serve_stats_by_surface(data)


def plot_head_to_head_rank(player1_data, player2_data, player1_name, player2_name, category):
    """
    Plots two players against eachother.

    Args:
        player1_data: Player 1's data.
        player2_data: Player 2's data.
        player1_name: The name of Player 1.
        player2_name: The name of Player 2.
        category: The chosen category.

    Returns:
        plot_html: HTML representation of graph.
        None: no graph if invalid data.
    """
    rank_column = 'ATP Rank' if category == 'LTA' else 'WTA Rank'

    if 'year-end-rankings' in player1_data and 'year-end-rankings' in player2_data:
        headers = ['Year']

        rankings_df1 = pd.DataFrame(player1_data['year-end-rankings']['data'], columns=player1_data['year-end-rankings']['headers'])
        rankings_df1.columns = [col.replace('\n', ' ') for col in rankings_df1.columns]
        rankings_df1['Year'] = rankings_df1['Year'].str.extract(r'(\d{4})')
        rankings_df1['Year'] = pd.to_numeric(rankings_df1['Year'], errors='coerce')
        rankings_df1[rank_column] = pd.to_numeric(rankings_df1[rank_column], errors='coerce')
        rankings_df1 = rankings_df1[['Year', rank_column]]
        rankings_df1.dropna(subset=headers, inplace=True)

        rankings_df2 = pd.DataFrame(player2_data['year-end-rankings']['data'], columns=player2_data['year-end-rankings']['headers'])
        rankings_df2.columns = [col.replace('\n', ' ') for col in rankings_df2.columns]
        rankings_df2['Year'] = rankings_df2['Year'].str.extract(r'(\d{4})')
        rankings_df2['Year'] = pd.to_numeric(rankings_df2['Year'], errors='coerce')
        rankings_df2[rank_column] = pd.to_numeric(rankings_df2[rank_column], errors='coerce')
        rankings_df2 = rankings_df2[['Year', rank_column]]
        rankings_df2.dropna(subset=headers, inplace=True)
    else:
        return None

    if 'tour-years' in player1_data:
        tour_years_df1 = pd.DataFrame(player1_data['tour-years']['data'], columns=player1_data['tour-years']['headers'])
    elif 'chall-years' in player1_data:
        tour_years_df1 = pd.DataFrame(player1_data['chall-years']['data'], columns=player1_data['chall-years']['headers'])
    else:
        return None

    if 'tour-years' in player2_data:
        tour_years_df2 = pd.DataFrame(player2_data['tour-years']['data'], columns=player2_data['tour-years']['headers'])
    elif 'chall-years' in player2_data:
        tour_years_df2 = pd.DataFrame(player2_data['chall-years']['data'], columns=player2_data['chall-years']['headers'])
    else:
        return None

    headers = ['Year', 'Win%']

    tour_years_df1.columns = [col.replace('\n', ' ') for col in tour_years_df1.columns]
    tour_years_df1['Year'] = tour_years_df1['Year'].str.extract(r'(\d{4})')
    tour_years_df1['Year'] = pd.to_numeric(tour_years_df1['Year'], errors='coerce')
    tour_years_df1['Win%'] = tour_years_df1['Win%'].str.replace('%', '').astype(float)
    tour_years_df1 = tour_years_df1[headers]
    tour_years_df1.dropna(subset=headers, inplace=True)

    tour_years_df2.columns = [col.replace('\n', ' ') for col in tour_years_df2.columns]
    tour_years_df2['Year'] = tour_years_df2['Year'].str.extract(r'(\d{4})')
    tour_years_df2['Year'] = pd.to_numeric(tour_years_df2['Year'], errors='coerce')
    tour_years_df2['Win%'] = tour_years_df2['Win%'].str.replace('%', '').astype(float)
    tour_years_df2 = tour_years_df2[headers]
    tour_years_df2.dropna(subset=headers, inplace=True)

    merged_rankings_df1 = rankings_df1[['Year', rank_column]].copy()
    merged_rankings_df1['Player'] = player1_name
    merged_rankings_df2 = rankings_df2[['Year', rank_column]].copy()
    merged_rankings_df2['Player'] = player2_name

    merged_win_df1 = tour_years_df1[['Year', 'Win%']].copy()
    merged_win_df1['Player'] = player1_name
    merged_win_df2 = tour_years_df2[['Year', 'Win%']].copy()
    merged_win_df2['Player'] = player2_name

    combined_rankings_df = pd.concat([merged_rankings_df1, merged_rankings_df2]).sort_values(by=['Year', rank_column])
    combined_win_df = pd.concat([merged_win_df1, merged_win_df2]).sort_values(by=['Year'])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=combined_rankings_df[combined_rankings_df['Player'] == player1_name]['Year'],
        y=combined_rankings_df[combined_rankings_df['Player'] == player1_name][rank_column],
        name=f'{player1_name} {rank_column}',
        marker=dict(color='rgba(207, 102, 121, 0.5)', line=dict(width=0)),
        width=0.4,
        offsetgroup=0,
        yaxis='y2',
        hovertemplate='Year: %{x}<br>Rank: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Bar(
        x=combined_rankings_df[combined_rankings_df['Player'] == player2_name]['Year'],
        y=combined_rankings_df[combined_rankings_df['Player'] == player2_name][rank_column],
        name=f'{player2_name} {rank_column}',
        marker=dict(color='rgba(134, 139, 252, 0.5)', line=dict(width=0)),
        width=0.4,
        offsetgroup=1,
        yaxis='y2',
        hovertemplate='Year: %{x}<br>Rank: %{y}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=combined_win_df[combined_win_df['Player'] == player1_name]['Year'],
        y=combined_win_df[combined_win_df['Player'] == player1_name]['Win%'],
        mode='lines+markers',
        name=f'{player1_name} Win%',
        line=dict(color='rgba(207, 102, 121, 0.9)'),
        hovertemplate='Year: %{x}<br>Win%: %{y}%<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=combined_win_df[combined_win_df['Player'] == player2_name]['Year'],
        y=combined_win_df[combined_win_df['Player'] == player2_name]['Win%'],
        mode='lines+markers',
        name=f'{player2_name} Win%',
        line=dict(color='rgba(134, 139, 252, 0.9)'),
        hovertemplate='Year: %{x}<br>Win%: %{y}%<extra></extra>' 
    ))

    max_rank = max(combined_rankings_df[rank_column].max(), rankings_df1[rank_column].max(), rankings_df2[rank_column].max())
    tickvals = []
    current = 1

    if max_rank >= 500:
        while current <= int(max_rank):
            tickvals.append(current)
            if current == 1:
                current = 200
            else:
                current += 200
    elif max_rank >= 250:
        while current <= int(max_rank):
            tickvals.append(current)
            if current == 1:
                current = 100
            else:
                current += 100
    else:
        while current <= int(max_rank):
            tickvals.append(current)
            if current == 1:
                current = 10
            else:
                current += 10

    ticktext = [str(val) for val in tickvals]

    fig.update_layout(
        title='Player Performance Comparison Over Time',
        xaxis_title='Year',
        xaxis=dict(
            gridcolor='#E0E0E0',
            linecolor='#E0E0E0',
            tickmode='linear'
        ),
        yaxis=dict(
            title='Win Percentage',
            range=[0, 100],
            gridcolor='#E0E0E0',
            linecolor='#E0E0E0',
            zeroline=False
        ),
        yaxis2=dict(
            title=f'{category} Rank',
            overlaying='y',
            side='right',
            range=[0, max_rank + 10],
            tickvals=tickvals,
            ticktext=ticktext,
            tickmode='array',
            showgrid=False,
            linecolor='#E0E0E0'
        ),
        barmode='group',
        plot_bgcolor='#202020',
        paper_bgcolor='#202020',
        font=dict(color='#E0E0E0'),
        legend=dict(
            title='Metrics',
            orientation='h',
            yanchor='bottom',
            xanchor='right',
            y=-0.2,
            x=1.1
        ),
        height=720,
        width=1200
    )

    plot_html = pio.to_html(fig, full_html=False)

    return plot_html

def plot_head_to_head_serves(player1_data, player2_data, player1_name, player2_name):
    """
    Plots two player's serve performance.

    Args:
        player1_data: Player 1's data.
        player2_data: Player 2's data.
        player1_name: The name of Player 1.
        player2_name: The name of Player 2.

    Returns:
        plot_html: HTML representation of graph.
        None: no graph if invalid data.
    """
    if 'tour-years' in player1_data:
        df1 = pd.DataFrame(player1_data['tour-years']['data'], columns=player1_data['tour-years']['headers'])
    elif 'chall-years' in player1_data:
        df1 = pd.DataFrame(player1_data['chall-years']['data'], columns=player1_data['chall-years']['headers'])
    else:
        return None

    if 'tour-years' in player2_data:
        df2 = pd.DataFrame(player2_data['tour-years']['data'], columns=player2_data['tour-years']['headers'])
    elif 'chall-years' in player2_data:
        df2 = pd.DataFrame(player2_data['chall-years']['data'], columns=player2_data['chall-years']['headers'])
    else:
        return None
    
    headers = ['1stIn', '1st%', 'A%', '2nd%']
    df1.replace('', pd.NA, inplace=True)
    df2.replace('', pd.NA, inplace=True)
    df1.dropna(subset=headers, inplace=True)
    df2.dropna(subset=headers, inplace=True)
    df1 = df1[headers].head(1)
    df2 = df2[headers].head(1)

    for head in headers:
        df1[head] = pd.to_numeric(df1[head].astype(str).str.replace('%', '').astype(float), errors='coerce')
        df2[head] = pd.to_numeric(df2[head].astype(str).str.replace('%', '').astype(float), errors='coerce')

    df1 = df1.iloc[0].values.flatten().tolist()
    df2 = df2.iloc[0].values.flatten().tolist()

    df1 += df1[:1]
    df2 += df2[:1]
    headers += headers[:1]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=df1,
        theta=headers,
        fill='toself',
        name=player1_name,
        hovertemplate = '<b>%{theta}</b>: %{r:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatterpolar(
        r=df2,
        theta=headers,
        fill='toself',
        name=player2_name,
        hovertemplate = '<b>%{theta}</b>: %{r:.2f}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                visible=True,
                linecolor='#202020',
                gridcolor='#202020'
            ),
            radialaxis=dict(
                visible=True,
                linecolor='#202020',
                gridcolor='#202020', 
            )
        ),
        showlegend=True,
        title=f'Serve Performance Comparison: {player1_name} vs {player2_name}',
        height=720,
        width=1200,
        plot_bgcolor='#202020',
        paper_bgcolor='#202020',
        font=dict(color='#E0E0E0')
    )

    plot_html = pio.to_html(fig, full_html=False)
    
    return plot_html

def plot_head_to_head_dominance(player1_data, player2_data, player1_name, player2_name):
    """
    Plots two player's dominance ratio.

    Args:
        player1_data: Player 1's  data.
        player2_data: Player 2's  data.
        player1_name: The name of Player 1.
        player2_name: The name of Player 2.

    Returns:
        plot_html: HTML representation of graph.
        None: no graph if invalid data.
    """
    if 'matches' in player1_data and 'matches' in player2_data:
        df1 = pd.DataFrame(player1_data['matches']['data'], columns=player1_data['matches']['headers'])
        df1.columns = [col.replace('\n', ' ') for col in df1.columns] 
        df2 = pd.DataFrame(player2_data['matches']['data'], columns=player2_data['matches']['headers'])
        df2.columns = [col.replace('\n', ' ') for col in df2.columns] 
        headers = ['vRk', 'DR']
        df1 = df1[headers]
        df2 = df2[headers]
        df1.replace('', pd.NA, inplace=True)
        df2.replace('', pd.NA, inplace=True)
        df1.dropna(subset=headers, inplace=True)
        df2.dropna(subset=headers, inplace=True)
        df1['DR'] = pd.to_numeric(df1['DR'], errors='coerce')
        df2['DR'] = pd.to_numeric(df2['DR'], errors='coerce')
        df1['vRk'] = pd.to_numeric(df1['vRk'], errors='coerce')
        df2['vRk'] = pd.to_numeric(df2['vRk'], errors='coerce')
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df1['DR'],
            y=df1['vRk'],
            mode='markers',
            name=f'{player1_name} Dominance Ratio',
            marker=dict(color='#868BFC', size=8),
            hovertemplate=(
                'DR: %{x}<br>'  
                'vRk: %{y}<br>'  
                '<extra></extra>'
            )
        ))

        if len(df1) > 1: 
            slope1, intercept1 = np.polyfit(df1['DR'], df1['vRk'], 1)
            fig.add_trace(go.Scatter(
                x=df1['DR'],
                y=slope1 * df1['DR'] + intercept1,
                mode='lines',
                name=f'{player1_name} Best Fit Line',
                line=dict(color='#868BFC')
                
            ))

        fig.add_trace(go.Scatter(
            x=df2['DR'],
            y=df2['vRk'],
            mode='markers',
            name=f'{player2_name} Dominance Ratio',
            marker=dict(color='#CF6679', size=8),
            hovertemplate=(
                'DR: %{x}<br>'  
                'vRk: %{y}<br>'  
                '<extra></extra>'
            )
        ))

        if len(df2) > 1:  
            slope2, intercept2 = np.polyfit(df2['DR'], df2['vRk'], 1)
            fig.add_trace(go.Scatter(
                x=df2['DR'],
                y=slope2 * df2['DR'] + intercept2,
                mode='lines',
                name=f'{player2_name} Best Fit Line',
                line=dict(color='#CF6679')
            ))

        fig.update_layout(
            title=f'{player1_name} vs {player2_name} - Dominance Ratio Comparison',
            xaxis_title='Dominance Ratio (DR)',
            yaxis_title='Opponent Ranking (vRk)',
            height=720,
            width=1200,
            plot_bgcolor='#202020',
            paper_bgcolor='#202020',
            font=dict(color='#E0E0E0'),
            xaxis=dict(
                gridcolor='#E0E0E0',  
                linecolor='#E0E0E0'   
            ),
            yaxis=dict(
                gridcolor='#E0E0E0',  
                linecolor='#E0E0E0',
                zeroline=False  
            )
        )

        plot_html = fig.to_html(full_html=False)
        
        return plot_html
    else:
        return None


def plot_data_h2h(player1_data, player2_data, player1_name, player2_name, category, plot_type='rank_h2h'):
    """
    Plot graph based off of selected plot type.

    Args:
        player1_data: Player 1's data.
        player2_data: Player 2's data.
        player1_name: The name of Player 1.
        player2_name: The name of Player 2.
        category: The category.
        plot_type: Type of plot to create.

    Returns:
        plot_html: HTML representation of graph.
        None: no graph if invalid data.
    """
    if plot_type == 'rank_h2h':
        return plot_head_to_head_rank(player1_data, player2_data, player1_name, player2_name, category)
    elif plot_type == 'radar_h2h':
        return plot_head_to_head_serves(player1_data, player2_data, player1_name, player2_name)
    elif plot_type == 'scatter_h2h':
        return plot_head_to_head_dominance(player1_data, player2_data, player1_name, player2_name)



