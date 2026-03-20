from youtubeSummary import YoutubeSummary
import streamlit as st


def main():
    st.header("Research Tool")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    video_id = st.text_input("Enter video id")
    
    if "video_id" not in st.session_state or st.session_state.video_id != video_id:
        st.session_state.video_id = video_id
        st.session_state.yt_summary = YoutubeSummary(video_id)
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the video"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            if video_id:
                with st.spinner("Analyzing video..."):
                    response = st.session_state.yt_summary.get_response(prompt)
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.warning("Please enter a video ID first.")


if __name__ == "__main__":
    main()
