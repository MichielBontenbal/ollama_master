"""
Dutch Text Simplifier - Compare Two LLM Models
Uses Ollama as backend with gemma3:1b and leesplank-noot-eurollm-1.7b models
"""

import streamlit as st
import requests
import json

# Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Default input text
DEFAULT_TEXT = """Een goede gezondheid vormt de fundering voor een actief en bevredigend bestaan. Het is niet alleen de afwezigheid van ziekte, maar een toestand van lichamelijk, geestelijk en sociaal welzijn. Een gebalanceerd dieet, rijk aan groenten, fruit, volle granen en magere eiwitten, voorziet het lichaam van essentiële voedingsstoffen. Regelmatige lichaamsbeweging, zoals wandelen, fietsen of krachttraining, versterkt niet alleen spieren en botten, maar vermindert ook stress en verbetert de stemming."""


def get_available_models() -> list[str]:
    """
    Fetch available models from Ollama.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model.get("name", "") for model in models if model.get("name")]
        return []
    except:
        return []


def simplify_text_with_ollama(text: str, model: str) -> str:
    """
    Send text to Ollama for simplification using the specified model.
    """
    prompt = f"""Je bent een expert in het vereenvoudigen van Nederlandse teksten. 
Vereenvoudig de volgende tekst zodat deze gemakkelijk te begrijpen is voor mensen met een laag leesniveau.
Gebruik korte zinnen, eenvoudige woorden, en vermijd moeilijke begrippen. Je geeft alleen de output tekst terug

Originele tekst:
{text}

Vereenvoudigde tekst:"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
            "stop": []  # ensure no accidental immediate stop
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "Geen antwoord ontvangen van het model.")
    except requests.exceptions.ConnectionError:
        return "❌ Fout: Kan geen verbinding maken met Ollama. Zorg ervoor dat Ollama draait op localhost:11434"
    except requests.exceptions.Timeout:
        return "❌ Fout: De aanvraag duurde te lang. Probeer het opnieuw."
    except requests.exceptions.RequestException as e:
        return f"❌ Fout bij communicatie met Ollama: {str(e)}"
    except json.JSONDecodeError:
        return "❌ Fout: Ongeldig antwoord van Ollama."


def main():
    # Page configuration
    st.set_page_config(
        page_title="Vereenvoudig Nederlandse teksten",
        page_icon="📝",
        layout="wide"
    )

    # Title and description
    st.title("📝 Tekstversimpeling Arena")
    st.markdown("*Vergelijk twee LLM-modellen voor het versimpelen van Nederlandse teksten*")
    
    st.divider()

    # Fetch available models
    available_models = get_available_models()
    
    if not available_models:
        st.warning("⚠️ Geen modellen gevonden. Zorg dat Ollama draait en modellen zijn geïnstalleerd.")
        available_models = ["Geen modellen beschikbaar"]

    # Input panel
    st.subheader("📥 Invoertekst")
    input_text = st.text_area(
        label="Voer de Nederlandse tekst in die je wilt vereenvoudigen:",
        value=DEFAULT_TEXT,
        height=200,
        help="Plak hier de tekst die je wilt laten vereenvoudigen door beide modellen."
    )

    # Simplify button
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        simplify_button = st.button(
            "🚀 Vereenvoudig Tekst",
            type="primary",
            use_container_width=True
        )

    st.divider()

    # Output panels - side by side
    st.subheader("📤 Resultaten")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model 1")
        selected_model_1 = st.selectbox(
            "Selecteer model 1:",
            options=available_models,
            index=0,
            key="model_1_select"
        )
        output_container_1 = st.container(border=True)
        
    with col2:
        st.markdown("### Model 2")
        # Default to second model if available, otherwise first
        default_index_2 = 1 if len(available_models) > 1 else 0
        selected_model_2 = st.selectbox(
            "Selecteer model 2:",
            options=available_models,
            index=default_index_2,
            key="model_2_select"
        )
        output_container_2 = st.container(border=True)

    # Process when button is clicked
    if simplify_button:
        if not input_text.strip():
            st.error("⚠️ Voer eerst een tekst in om te vereenvoudigen.")
        elif available_models[0] == "Geen modellen beschikbaar":
            st.error("⚠️ Geen modellen beschikbaar. Start Ollama en installeer modellen.")
        else:
            with col1:
                with output_container_1:
                    with st.spinner(f"Model 1 ({selected_model_1}) is bezig..."):
                        result_1 = simplify_text_with_ollama(input_text, selected_model_1)
                    st.markdown(result_1)

            with col2:
                with output_container_2:
                    with st.spinner(f"Model 2 ({selected_model_2}) is bezig..."):
                        result_2 = simplify_text_with_ollama(input_text, selected_model_2)
                    st.markdown(result_2)

            # Show comparison stats
            st.divider()
            st.subheader("📊 Vergelijking")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric(
                    label="Originele tekst",
                    value=f"{len(input_text.split())} woorden"
                )
            
            with stat_col2:
                if not result_1.startswith("❌"):
                    st.metric(
                        label=f"Model 1 ({selected_model_1})",
                        value=f"{len(result_1.split())} woorden"
                    )
                else:
                    st.metric(label=f"Model 1 ({selected_model_1})", value="Fout")
            
            with stat_col3:
                if not result_2.startswith("❌"):
                    st.metric(
                        label=f"Model 2 ({selected_model_2})",
                        value=f"{len(result_2.split())} woorden"
                    )
                else:
                    st.metric(label=f"Model 2 ({selected_model_2})", value="Fout")
    else:
        # Show placeholder text when not yet processed
        with output_container_1:
            st.markdown("*Klik op 'Vereenvoudig Tekst' om het resultaat te zien...*")
        with output_container_2:
            st.markdown("*Klik op 'Vereenvoudig Tekst' om het resultaat te zien...*")

    # Sidebar with instructions
    with st.sidebar:
        st.header("ℹ️ Instructies")
        st.markdown("""
        **Vereisten:**
        1. Zorg dat [Ollama](https://ollama.ai) is geïnstalleerd en draait
        2. Download de benodigde modellen:
        
        ```bash
        ollama pull <modelname>
        ```
        
        **Gebruik:**
        1. Selecteer de modellen die je wilt vergelijken
        2. Voer je Nederlandse tekst in
        3. Klik op 'Vereenvoudig Tekst'
        4. Vergelijk de resultaten van beide modellen
        """)
        
        st.divider()
        
        st.header("⚙️ Status")
        # Check Ollama connection
        if available_models and available_models[0] != "Geen modellen beschikbaar":
            st.success("✅ Ollama is verbonden")
            st.markdown("**Beschikbare modellen:**")
            for model in available_models:
                st.markdown(f"- `{model}`")
        else:
            st.error("❌ Kan niet verbinden met Ollama")
            st.markdown("Start Ollama met: `ollama serve`")


if __name__ == "__main__":
    main()
