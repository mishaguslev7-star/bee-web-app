with col_res:
    if file and model:
        # Логика правильного склонения слова "особь"
        def get_bee_word(n):
            if 11 <= n % 100 <= 19:
                return "особей"
            last_digit = n % 10
            if last_digit == 1:
                return "особь"
            if 2 <= last_digit <= 4:
                return "особи"
            return "особей"

        bee_text = get_bee_word(count)
        st.metric(f"ОБНАРУЖЕНО", f"{count} {bee_text}")
        
        st.markdown(f"""
        <div class="info-card">
            <h3>📊 Текущий отчёт</h3>
            На снимке зафиксировано <b>{count} {bee_text}</b>. 
            Это позволяет оценить активность летка без прямого вмешательства в жизнь улья.
        </div>
        """, unsafe_allow_html=True)
