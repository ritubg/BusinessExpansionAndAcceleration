import streamlit as st
import requests
import pandas as pd
from PIL import Image
from textblob import TextBlob
from bs4 import BeautifulSoup
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

API_2A = "http://127.0.0.1:8000"  # API 2A
API_2B = "http://127.0.0.1:8001"  # API 2B 
API_2CD = "http://127.0.0.1:8002"  # API 2CD
API_3A = "http://127.0.0.1:8003" #API 3A
API_3B = "http://127.0.0.1:8004" #API 3B
API_4A = "http://127.0.0.1:8005" #API 4A


def get_google_news_headlines(query, max_headlines=15):
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    headlines = []
    for h in soup.find_all('h3'):
        text = h.get_text().strip()
        if len(text) > 10:
            headlines.append(text)
        if len(headlines) >= max_headlines:
            break
    return headlines

def analyze_sentiment(headlines):
    results = []
    for h in headlines:
        polarity = TextBlob(h).sentiment.polarity
        sentiment_type = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        results.append({'headline': h, 'polarity': polarity, 'sentiment': sentiment_type})
    return pd.DataFrame(results)

def aggregate_sentiment(df):
    polarity_avg = df['polarity'].mean()
    overall = "Positive" if polarity_avg > 0 else "Negative" if polarity_avg < 0 else "Neutral"
    return overall, polarity_avg



st.title("Business Expansion and Acceralation")

dashboard = st.sidebar.selectbox("Select ", ["Dashboard 1", "Execution Intelligence", "Client Intelligence", "Financial Intelligence", "Dashboard 5"])

if dashboard == "Execution Intelligence":
    st.header("Cost Overrun & Delay Predictor for Existing Projects")

    try:
        report = requests.get(f"{API_2A}/full_report").json()

        st.subheader("Insights")
        for k, v in report.get("insights", {}).items():
            st.write(f"{k.replace('_', ' ').title()}: {v}")

        st.subheader("Top Risky Projects")
        st.dataframe(pd.DataFrame(report.get("top_risky_projects", [])))

        st.subheader("Top Delay Projects")
        st.dataframe(pd.DataFrame(report.get("top_delay_projects", [])))

        st.subheader("At-Risk Projects (Combined)")
        st.dataframe(pd.DataFrame(report.get("at_risk_projects", [])))
        st.subheader("Recommendations")
        for rec in report.get("recommendations", {}).get("recommendations", []):
            st.info(f"[{rec['priority']}] {rec['category']}: {rec['message']}")

        for plot_name, plot_path in report.get("plot_paths", {}).items():
            st.image(f"{API_2A}/plot/{plot_name}", caption=plot_name)


    except Exception as e:
        st.error(f"Failed to fetch 2A results: {e}")

    st.markdown("---")
    st.header("Project Cost Estimation & Timeline for New Projects")

    try:
        metrics_2b = requests.get(f"{API_2B}/metrics").json()
       
        top_risky_2b = requests.get(f"{API_2B}/top_risky_cost").json()
        if isinstance(top_risky_2b, list) and len(top_risky_2b) > 0:
            st.subheader("Top Risky Projects (2B)")
            st.dataframe(pd.DataFrame(top_risky_2b))

        safe_projects_2b = requests.get(f"{API_2B}/safe_projects_cost").json()
        if isinstance(safe_projects_2b, list) and len(safe_projects_2b) > 0:
            st.subheader("Safe Projects (2B)")
            st.dataframe(pd.DataFrame(safe_projects_2b))
            
        plots_2b = [
            "2b_cost_predictions",
            "2b_timeline_predictions",
            "2b_cost_overrun_dist",
            "2b_timeline_overrun_dist",
            "2b_cost_scatter",
            "2b_timeline_scatter",
            
        ]

        for plot_name in plots_2b:
            try:
                plot_url = f"{API_2B}/plot/{plot_name}"
                st.image(plot_url, caption=plot_name.replace("2b_", "").replace("_", " ").title(), use_container_width=True)
            except Exception as e:
                st.warning(f" Could not load plot: {plot_name} — {e}")

    except Exception as e:
        st.error(f"Failed to fetch 2B results: {e}")

    st.header("Scenario Planning and new project takeover decision")

    try:
        decisions = requests.get(f"{API_2CD}/decisions").json()
        if isinstance(decisions, list) and len(decisions) > 0:
            st.subheader("Takeover Decisions per Project")
            st.dataframe(pd.DataFrame(decisions))
    except Exception as e:
        st.error(f"Failed to fetch takeover decisions: {e}")

    st.markdown("---")

    try:
        compare = requests.get(f"{API_2CD}/compare").json()
        if isinstance(compare, list) and len(compare) > 0:
            st.subheader("Model A vs Model B Stress Scores & Vulnerability")
            st.dataframe(pd.DataFrame(compare))
    except Exception as e:
        st.error(f"Failed to fetch model comparison: {e}")

    st.markdown("---")

    try:
        response = requests.get(f"{API_2CD}/insights").json()
        if response.get("status") == "success":
            insights = response["insights"]
            st.subheader("Key Insights from Portfolio Analysis")
            
            st.json(insights)
        else:
            st.error(f"Error fetching insights: {response.get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to fetch insights: {e}")
    st.markdown("---")


    try:
        summary = requests.get(f"{API_2CD}/summary").json()
        if summary.get("status") == "success":
            st.subheader("Portfolio Summary")
            st.metric("Total Projects", summary.get("total_projects", 0))
            st.metric("Average Stress Score", round(summary.get("average_stress_score", 0), 3))
            st.metric("Average Vulnerability", round(summary.get("average_vulnerability", 0), 3))
            
            st.write("Decision Breakdown:")
            breakdown = summary.get("decision_breakdown", {})
            st.json(breakdown)
    except Exception as e:
        st.error(f"Failed to fetch summary metrics: {e}")
    plot_endpoints = {
    "Correlation Heatmap": f"{API_2CD}/plot/correlation_heatmap",
    "Cost vs Duration": f"{API_2CD}/plot/cost_vs_duration",
    "Feature Relationships": f"{API_2CD}/plot/pairplots_features"
    }

    for title, endpoint in plot_endpoints.items():
        try:
            plot_response = requests.get(endpoint)
            if plot_response.status_code == 200:
                image = Image.open(BytesIO(plot_response.content))
                st.image(image, caption=title, use_column_width=True)
            else:
                st.warning(f"{title} not available — ensure {endpoint.split('/')[-1]}.png exists in artifacts.")
        except Exception as e:
            st.error(f"Failed to load {title}: {e}")

elif dashboard == "Client Intelligence":
    st.header("Churn Prediction ")

    try:
        metrics = requests.get(f"{API_3A}/metrics").json()
    except Exception as e:
        st.error(f"Failed to fetch 3A metrics: {e}")

    st.markdown("---")

 
    try:
        top_churn = requests.get(f"{API_3A}/top_churn?n=10").json()
        if isinstance(top_churn, list) and len(top_churn) > 0:
            st.subheader("Top 10 Clients with Highest Churn Probability")
            st.dataframe(pd.DataFrame(top_churn))
    except Exception as e:
        st.error(f"Failed to fetch top churn clients: {e}")

    st.markdown("---")

    try:
        report = requests.get(f"{API_3A}/report").json()
        if isinstance(report, list) and len(report) > 0:
            st.subheader("Full Client Churn Report")
            st.json(report)
    except Exception as e:
        st.error(f"Failed to fetch full client report: {e}")

    plot_names = {
    "churn_distribution": "Churn Distribution",
    "value_vs_churn": "Client Value vs Churn Probability",
    "action_summary": "Recommended Actions Summary",
    "feature_heatmap": "Feature Correlation Heatmap"
    }

    try:
        for key, title in plot_names.items():
            try:
                response = requests.get(f"{API_3A}/plots/{key}")
                if response.status_code == 200:
                    st.image(response.content, caption=title, use_container_width=True)
                else:
                    st.warning(f"{title} plot not found on server.")
            except Exception as e:
                st.error(f"Failed to load {title}: {e}")
    except Exception as e:
        st.error(f"Failed to fetch churn analysis plots: {e}")


    st.header("Client Segmentation")

    st.markdown("---")


    try:
        summary = requests.get(f"{API_3B}/summary").json()
        if isinstance(summary, list) and len(summary) > 0:
            st.subheader("Cluster Summary")
            st.dataframe(pd.DataFrame(summary))
    except Exception as e:
        st.error(f"Failed to fetch cluster summary: {e}")

    st.markdown("---")


    try:
        clusters = requests.get(f"{API_3B}/clusters").json()
        if isinstance(clusters, list) and len(clusters) > 0:
            st.subheader("All Projects with Cluster Assignment")
            st.dataframe(pd.DataFrame(clusters))
    except Exception as e:
        st.error(f"Failed to fetch clusters: {e}")

    
    plot_names = {
    "avg_profitability_per_cluster": "Average Profitability per cluster",
    "cluster_count": "Cluster county",
    "cluster_scatter": "Cluster scatter",
    "feature_heatmap": "Feature Correlation Heatmap",
    "profitability_boxplot": "Profitability Boxplot"
    }

    try:
        for key, title in plot_names.items():
            try:
                response = requests.get(f"{API_3B}/plots/{key}")
                if response.status_code == 200:
                    st.image(response.content, caption=title, use_container_width=True)
                else:
                    st.warning(f"{title} plot not found on server.")
            except Exception as e:
                st.error(f"Failed to load {title}: {e}")
    except Exception as e:
        st.error(f"Failed to fetch churn analysis plots: {e}")

    st.markdown("---")

elif dashboard == "Financial Intelligence":
    
    st.title("Revenue & Cashflow Forecasting")

    try:
        response = requests.get(f"{API_4A}/metrics").json()
        
        if response.get("status") == "success":
            st.subheader("Model Metrics")
            st.markdown(f"**Best Model:** {response.get('best_model_name', 'N/A')}")

            def format_metric(val):
                """Format metric: use 4 decimals if >0.001, scientific if smaller."""
                if abs(val) >= 0.001:
                    return f"{val:.4f}"
                else:
                    return f"{val:.2e}"

            metrics = response.get("metrics", {})
            for model, m in metrics.items():
                st.markdown(f"**{model} Metrics:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", format_metric(m.get("MAE", 0)))
                col2.metric("RMSE", format_metric(m.get("RMSE", 0)))
                col3.metric("MAPE", format_metric(m.get("MAPE", 0)))
                st.markdown("---")
        else:
            st.error(f"Error fetching 4A metrics: {response.get('error', 'Unknown error')}")
            
    except Exception as e:
        st.error(f"Failed to fetch 4A metrics: {e}")


    st.markdown("---")

    try:
        rev_plot_url = f"{API_4A}/plots/revenue"
        st.subheader("Revenue Forecast")
        img = Image.open(requests.get(rev_plot_url, stream=True).raw)
        st.image(img, use_column_width=True)
    except Exception as e:
        st.error(f"Failed to fetch revenue plot: {e}")

    st.markdown("---")

    try:
        liq_plot_url = f"{API_4A}/plots/liquidity"
        st.subheader("Liquidity Forecast")
        img = Image.open(requests.get(liq_plot_url, stream=True).raw)
        st.image(img, use_column_width=True)
    except Exception as e:
        st.error(f"Failed to fetch liquidity plot: {e}")

    st.markdown("---")

    try:
        forecast = requests.get(f"{API_4A}/forecast").json()
        if isinstance(forecast, dict) and len(forecast) > 0:
            st.subheader("Forecasted Revenue & Liquidity")
            df_forecast = pd.DataFrame({
                "Month": forecast["future_dates"],
                "Revenue_Forecast_INR": forecast["future_pred"],
                "Liquidity_Forecast_INR": forecast["liquidity_forecast"]
            })
            st.dataframe(df_forecast)
        else:
            st.error("No forecast data available")
    except Exception as e:
        st.error(f"Failed to fetch forecast table: {e}")

    st.subheader("Market & Investor Sentiment Analysis")
    company_query = "Salarpuria Sattva Group"
    industry_queries = [
        "Indian real estate sector policy news",
        "Construction industry India updates",
        "Housing sector growth and regulations"
    ]

    company_headlines = get_google_news_headlines(company_query)
    company_sentiment_df = analyze_sentiment(company_headlines)
    company_overall, company_score = aggregate_sentiment(company_sentiment_df)

    industry_headlines = []
    for q in industry_queries:
        industry_headlines += get_google_news_headlines(q, max_headlines=10)
    industry_sentiment_df = analyze_sentiment(industry_headlines)
    industry_overall, industry_score = aggregate_sentiment(industry_sentiment_df)

    st.markdown(f"**Company ({company_query}) Sentiment:** {company_overall} ({company_score:.3f})")
    st.markdown(f"**Industry (Construction Sector) Sentiment:** {industry_overall} ({industry_score:.3f})")

    st.bar_chart(pd.DataFrame({
        "Score": [company_score, industry_score]
    }, index=["Company", "Industry"]))

    st.subheader("Recent Headlines")
    st.markdown("**Company Headlines:**")
    st.dataframe(company_sentiment_df[['headline','sentiment','polarity']])
    st.markdown("**Industry Headlines:**")
    st.dataframe(industry_sentiment_df[['headline','sentiment','polarity']])

    plt.figure(figsize=(6,4))
    sns.countplot(x='sentiment', data=company_sentiment_df, color='skyblue', order=['Positive','Neutral','Negative'])
    plt.title("Sentiment Distribution of Headlines")
    plt.ylabel("Number of Headlines")
    plt.xlabel("Sentiment")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(plt.gcf())
    plt.close()

    plt.figure(figsize=(6,4))
    sns.histplot(company_sentiment_df['polarity'], bins=10, kde=True, color='skyblue')
    plt.title("Polarity Score Distribution")
    plt.xlabel("Polarity")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(plt.gcf())
    plt.close()