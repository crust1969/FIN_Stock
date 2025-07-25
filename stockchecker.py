import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import smtplib
from email.mime.text import MIMEText
import plotly.graph_objects as go
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

# 📬 E-Mail Alert senden
def send_email_alert(subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = st.secrets["EMAIL_USER"]
        msg["To"] = st.secrets["EMAIL_TO"]

        with smtplib.SMTP(st.secrets["EMAIL_HOST"], st.secrets["EMAIL_PORT"]) as server:
            server.starttls()
            server.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
            server.sendmail(msg["From"], [msg["To"]], msg.as_string())
        st.success("📧 Alert-E-Mail wurde gesendet!")
    except Exception as e:
        st.error(f"Fehler beim E-Mail-Versand: {e}")

# 📈 Plotly-Chart
def show_chart(symbol):
    df = yf.Ticker(symbol).history(period="6mo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Kurs"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="SMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), name="SMA50"))
    fig.update_layout(title=f"{symbol} – Kurs mit SMA20 & SMA50", xaxis_title="Datum", yaxis_title="Preis (USD)")
    st.plotly_chart(fig)

# 🔧 Analysefunktionen
def get_price_data(symbol: str) -> str:
    df = yf.Ticker(symbol).history(period="6mo")
    latest = df["Close"].iloc[-1]
    return f"Aktueller Kurs von {symbol}: {latest:.2f} USD"

def get_indicators(symbol: str) -> str:
    df = yf.Ticker(symbol).history(period="6mo")
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = round(rsi.iloc[-1], 1)

    short_ema = df["Close"].ewm(span=12, adjust=False).mean()
    long_ema = df["Close"].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()

    macd_now = macd.iloc[-1]
    signal_now = signal.iloc[-1]
    trend = "bullish" if macd_now > signal_now else "bearish"

    # Optionaler Alert
    if latest_rsi < 30:
        send_email_alert(
            subject=f"📉 RSI-Alarm für {symbol}",
            body=f"RSI von {symbol} liegt bei {latest_rsi} → möglicher Kaufzeitpunkt?"
        )

    return f"RSI: {latest_rsi} | MACD: {macd_now:.2f} vs Signal: {signal_now:.2f} → {trend}"

def get_peg(symbol: str) -> str:
    key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={key}"
    r = requests.get(url)
    if r.status_code == 200:
        peg = r.json().get("PEGRatio", "N/A")
        try:
            peg_val = float(peg)
            if peg_val < 1:
                send_email_alert(
                    subject=f"📊 PEG-Alarm für {symbol}",
                    body=f"PEG-Ratio von {symbol} = {peg} → möglich unterbewertet."
                )
        except:
            pass
        return f"PEG-Ratio von {symbol}: {peg}"
    return "Fehler beim PEG-Abruf"

# 🧠 Agent-Initialisierung
def create_agent():
    tools = [
        Tool(name="Kursdaten", func=get_price_data, description="Gibt aktuellen Kurs zurück"),
        Tool(name="Technische Analyse", func=get_indicators, description="RSI und MACD analysieren"),
        Tool(name="PEG-Ratio", func=get_peg, description="Gibt PEG-Ratio aus Fundamental-Daten zurück"),
    ]
    llm = ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

# 🖼️ UI
st.set_page_config(page_title="📊 Aktienanalyse Agent", layout="wide")
st.title("📈 Intelligente Aktienanalyse mit LangChain")

tickers = st.text_input("🔎 Ticker eingeben (z. B. AAPL, MSFT, AMZN):", "AAPL, MSFT")
frage = st.text_input("🧠 Frage an den Agenten:", "Analysiere die Aktie mit RSI, MACD und PEG")

if st.button("🚀 Analyse starten"):
    agent = create_agent()
    for symbol in [t.strip().upper() for t in tickers.split(",")]:
        st.header(f"🔍 {symbol}")
        result = agent.run(f"{frage.replace('die Aktie', symbol)}")
        st.markdown(result)
        show_chart(symbol)
