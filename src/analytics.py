import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def get_summary_metrics(df):
    """
    Calculates summary KPIs from dataset.
    """
    total_destinations = len(df)
    avg_daily_cost = df['total_daily_cost_inr'].mean() if 'total_daily_cost_inr' in df.columns else 0.0
    num_categories = df['destination_type'].nunique() if 'destination_type' in df.columns else 0
    avg_rating = df['popularity_score'].mean() if 'popularity_score' in df.columns else 0.0

    return {
        'total_destinations': total_destinations,
        'avg_daily_cost_inr': round(avg_daily_cost, 2),
        'num_categories': num_categories,
        'avg_rating': round(avg_rating, 2)
    }

def plot_top_expensive_destinations(df, top_n=10):
    """
    1. Interactive Plotly Bar chart of top_n destinations with highest estimated daily costs.
    """
    top_df = df.sort_values(by='total_daily_cost_inr', ascending=False).head(top_n).iloc[::-1]

    fig = px.bar(
        top_df,
        x='total_daily_cost_inr',
        y='destination',
        orientation='h',
        color='total_daily_cost_inr',
        color_continuous_scale=['#38BDF8', '#0284C7', '#0369A1', '#075985'],
        hover_data={'state': True, 'destination_type': True, 'total_daily_cost_inr': ':.2f', 'popularity_score': ':.2f'},
        labels={'total_daily_cost_inr': 'Daily Cost (INR ₹)', 'destination': 'Destination', 'state': 'State', 'destination_type': 'Category'}
    )

    fig.update_layout(
        title=dict(text=f"<b>Top {top_n} Most Expensive Destinations (Daily Cost ₹)</b>", font=dict(size=16, color="#0F172A")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=40),
        height=420,
        font=dict(color="#0F172A"),
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", tickprefix="₹", tickfont=dict(color="#0F172A")),
        yaxis=dict(showgrid=False, tickfont=dict(color="#0F172A")),
        coloraxis_showscale=False
    )
    fig.update_traces(
        texttemplate='₹%{x:,.0f}',
        textposition='outside',
        marker_line_color='rgba(0,0,0,0)',
        hovertemplate='<b>%{y}</b><br>Daily Cost: ₹%{x:,.2f}<extra></extra>'
    )
    return fig

def plot_cost_distribution(df):
    """
    2. Interactive Histogram showing daily travel cost distribution.
    """
    fig = px.histogram(
        df,
        x='total_daily_cost_inr',
        nbins=20,
        color_discrete_sequence=['#0284C7'],
        marginal="box",
        labels={'total_daily_cost_inr': 'Daily Cost (INR ₹)'}
    )

    mean_val = df['total_daily_cost_inr'].mean()
    median_val = df['total_daily_cost_inr'].median()

    fig.add_vline(x=mean_val, line_dash="dash", line_color="#DC2626", annotation_text=f"Mean: ₹{mean_val:,.0f}", annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="solid", line_color="#16A34A", annotation_text=f"Median: ₹{median_val:,.0f}", annotation_position="top left")

    fig.update_layout(
        title=dict(text="<b>Daily Travel Expense Distribution Across Destinations</b>", font=dict(size=16, color="#0F172A")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=40),
        height=400,
        font=dict(color="#0F172A"),
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", tickprefix="₹", tickfont=dict(color="#0F172A")),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", tickfont=dict(color="#0F172A")),
        showlegend=False
    )
    return fig

def plot_correlation_heatmap(df):
    """
    3. Interactive Heatmap of numerical feature correlations.
    """
    num_cols = [
        'average_hotel_cost', 'average_food_cost',
        'average_local_transport_cost', 'average_activity_cost',
        'total_daily_cost_inr', 'recommended_duration', 'popularity_score'
    ]

    available_cols = [c for c in num_cols if c in df.columns]
    corr_matrix = df[available_cols].corr().round(2)

    clean_labels = [
        col.replace('average_', '').replace('_cost', ' cost').replace('_inr', ' (₹)').replace('_', ' ').title()
        for col in available_cols
    ]

    fig = px.imshow(
        corr_matrix,
        x=clean_labels,
        y=clean_labels,
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto"
    )

    fig.update_layout(
        title=dict(text="<b>Numerical Feature Correlation Matrix</b>", font=dict(size=16, color="#0F172A")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=40),
        height=420,
        font=dict(color="#0F172A"),
        xaxis=dict(tickfont=dict(color="#0F172A")),
        yaxis=dict(tickfont=dict(color="#0F172A"))
    )
    return fig

def plot_destination_types_pie(df):
    """
    4. Interactive Donut Chart showing destination types distribution.
    """
    type_counts = df['destination_type'].value_counts().reset_index()
    type_counts.columns = ['Category', 'Count']

    fig = px.pie(
        type_counts,
        names='Category',
        values='Count',
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Prism
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
    )

    fig.update_layout(
        title=dict(text="<b>Proportion of Destinations by Travel Category</b>", font=dict(size=16, color="#0F172A")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=40),
        height=400,
        showlegend=True,
        font=dict(color="#0F172A"),
        legend=dict(font=dict(color="#0F172A"))
    )
    return fig
