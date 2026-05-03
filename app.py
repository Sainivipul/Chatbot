import streamlit as st
import pandas as pd
import ollama
import re
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="CSV AI Analyst",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Header
# -------------------------------
st.markdown("## 📊 CSV AI Analyst")
st.caption("Ask questions about your data using natural language")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# -------------------------------
# Clean LLM Output (ROBUST)
# -------------------------------
def clean_code(response_text):
    # Remove markdown blocks
    code = re.sub(r"```.*?```", "", response_text, flags=re.DOTALL)

    # Extract lines containing df
    lines = code.split("\n")
    valid_lines = [line.strip() for line in lines if "df" in line]

    if valid_lines:
        code = valid_lines[-1]
    else:
        code = code.strip()

    # Remove any .plot()
    code = re.sub(r"\.plot\(.*?\)", "", code)

    return code.strip()

# -------------------------------
# Main App
# -------------------------------
if uploaded_file is not None:
    st.success("✅ File uploaded successfully")

    df = pd.read_csv(uploaded_file)

    # Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

    with col2:
        st.subheader("📊 Dataset Info")
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.write(df.dtypes)

    # Chat input
    user_query = st.chat_input("💬 Ask something about your data")

    if user_query:
        st.chat_message("user").write(user_query)

        query_lower = user_query.lower()

        # Detect visualization intent (robust)
        is_plot = any(word in query_lower for word in [
            "plot", "chart", "graph", "bar", "line", "pie"
        ])

        # Strict prompt
        prompt = f"""
You are a strict pandas code generator.

DataFrame columns: {list(df.columns)}

STRICT RULES:
- Output ONLY valid Python pandas code
- Output EXACTLY ONE LINE
- DO NOT include explanations
- DO NOT include text before or after code
- DO NOT include comments
- DO NOT include .plot()
- Return only dataframe or series

DataFrame name is df

Query: {user_query}
"""

        # LLM call
        with st.spinner("🤖 Thinking..."):
            response = ollama.chat(
                model='llama3',
                messages=[{"role": "user", "content": prompt}]
            )

        raw_code = response['message']['content']
        code = clean_code(raw_code)

        with st.chat_message("assistant"):
            st.markdown("### ⚙️ Generated Code")
            st.code(code)

        # Validate code
        if not code.startswith("df"):
            st.error("⚠️ Invalid code generated. Try rephrasing your question.")
            st.stop()

        # Execute safely
        try:
            allowed_globals = {"df": df, "pd": pd}
            result = eval(code, {"__builtins__": {}}, allowed_globals)

            st.markdown("### 📊 Output")

            # -------------------------------
            # Visualization
            # -------------------------------
            if is_plot:
                fig, ax = plt.subplots()

                if isinstance(result, pd.DataFrame) and result.shape[1] >= 2:
                    x = result.iloc[:, 0]
                    y = result.iloc[:, 1]
                    ax.bar(x, y)

                elif isinstance(result, pd.Series):
                    result.plot(kind='bar', ax=ax)

                else:
                    st.warning("⚠️ Cannot plot this result type")

                ax.set_title("Generated Chart")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            else:
                # -------------------------------
                # Smart Result Display
                # -------------------------------
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result, use_container_width=True)

                elif isinstance(result, pd.Series):
                    st.dataframe(result.to_frame(), use_container_width=True)

                else:
                    st.metric(label="Result", value=result)

        except Exception as e:
            st.error(f"⚠️ Error executing query: {e}")

else:
    st.info("⬆️ Upload a CSV file to get started")
