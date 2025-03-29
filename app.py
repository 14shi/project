import os
import time
import torch
import base64
import requests
import streamlit as st
import streamlit_nested_layout
import streamlit_antd_components as sac

from PIL import Image
from io import BytesIO
from streamlit_image_select import image_select
from streamlit_option_menu import option_menu
from configs import (
    BATCH_SIZE,
    CHUNK_SIZE,
)

BASE_URL = "http://localhost:8000"

st.set_page_config("多模态检索系统", page_icon="💫", layout="wide")
state = st.session_state
torch.classes.__path__ = []

st.markdown(
    """<style>
.stDeployButton {
    visibility: hidden;
}
.stMainMenu {
    visibility: hidden;
}
.stSidebarHeader {
    height: 0rem;
}
.stAppHeader {
    height: 0rem;
}
.stMainBlockContainer {
    padding: 0rem 2rem 0rem 2rem;
}
.st-emotion-cache-1mi2ry5 {
    padding: 0rem 1rem;
}
.st-emotion-cache-1gwvy71 {
    padding: 1rem 1rem;
}
.class .e1nzilvr5 {
    padding: 0rem 0rem;
}
[data-testid="stDecoration"] {
    background-image: linear-gradient(90deg, rgb(255, 255, 255), rgb(255, 255, 255));
}
[data-testid="stHeader"] {
    height: 0rem;
}
[data-testid="stSidebarHeader"] {
    height: 0rem;
}
[data-testid="stAppViewBlockContainer"] {
    padding: 0rem 6rem 0rem 6rem;
}
[data-testid="stToolbar"] {
    visibility: hidden;
}
.stVerticalBlock st-emotion-cache-fvs0ay e1f1d6gn2 {
    gap: 0rem;
}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def cache_data():
    resp = requests.get(BASE_URL + "/get_collections")
    c_names = resp.json()
    data = {"id": [], "content": [], "type": [], "select": []}
    for name in c_names:
        resp = requests.get(BASE_URL + f"/get_collection_points?collection_name={name}")
        d = resp.json()
        for di in d:
            data["id"].append(di[0])
            data["content"].append(di[1])
            data["type"].append(name)
            data["select"].append(False)

    return data


def now_time():

    return f':green[> {time.strftime("%H:%M:%S")}] '


def init():
    _p = {
        "query": None,
        "query_type": "text",
        "result": {"image_list": [], "text_list": []},
        "enable_retrieve": False,
        "change_select_upload": False,
    }
    for k, v in _p.items():
        if k not in state:
            state[k] = v

    os.makedirs("database/files", exist_ok=True)
    os.makedirs("database/images", exist_ok=True)


def sidebar():
    with st.sidebar:

        st.markdown(
            "<h1 style='text-align: center; '>多模态检索系统</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        state["MENU"] = option_menu(
            "目录",
            ["图文检索", "数据库管理"],
            icons=["activity", "file-image", "database"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "background-color": "#fafafa"
                },  # "padding": "0!important",
                "icon": {"color": "orange", "font-size": "15px"},
                "nav-link": {
                    "font-size": "18px",
                    "text-align": "left",
                    "margin": "2px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#88D8B0"},
            },
            key="MENU_KEY",
        )

        if state.MENU == "图文检索":
            st.markdown("**检索设置**")
            with st.expander("点击展开", expanded=True):
                st.markdown("🔸 :rainbow[**检索数量**]")
                cols = st.columns([0.05, 0.9, 0.05])
                cols[1].slider(
                    "xxx",
                    value=8,
                    min_value=1,
                    max_value=30,
                    label_visibility="collapsed",
                    key="top_k_key",
                )
                st.markdown("🔸 :rainbow[**检索阈值**]")
                cols = st.columns([0.05, 0.45, 0.45, 0.05])
                cols[1].slider(
                    "**Image**",
                    value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    key="image_thres_key",
                )
                cols[2].slider(
                    "**Text**",
                    value=0.0,
                    min_value=0.0,
                    max_value=1.0,
                    key="text_thres_key",
                )

                st.markdown("🔸 :rainbow[**显示相关性**]")
                cols = st.columns([0.05, 0.9, 0.05])
                cols[1].radio(
                    "xxx",
                    ["显示", "不显示"],
                    label_visibility="collapsed",
                    horizontal=True,
                    key="show_score_key",
                )


def body():
    st.markdown(
        "<h3 style='text-align: center;'><a href='#' style='text-decoration: none'>多模态检索系统</a></h3>",
        unsafe_allow_html=True,
    )

    if state.MENU == "图文检索":

        with st.container(height=520):
            tab0_ph, tab1_ph = st.tabs(["**图片🔖**", "**文本📑**"])

        # query input
        if msg := st.chat_input(
            "输入文本/选择图片， Enter 进行检索",
            accept_file=True,
            file_type=["png", "jpg"],
        ):
            if file := msg["files"]:
                state.query = base64.b64encode(file[0].getvalue()).decode("utf-8")
                state.query_type = "image"
            else:
                state.query = msg["text"]
                state.query_type = "text"

            state.enable_retrieve = True

        # 查找数据
        if state.enable_retrieve:
            state.enable_retrieve = False
            with tab0_ph.container(), st.spinner("正在检索..."):
                payload = dict(
                    query=state.query,
                    query_type=state.query_type,
                    retrieve_type="full",
                    top_k=state.top_k_key,
                    image_thres=state.image_thres_key,
                    text_thres=state.text_thres_key,
                    use_llm_summary=False,
                )
                resp = requests.post(BASE_URL + "/retrieve_image_text", json=payload)
                retrieve_texts, retrieve_images = resp.json()
            state.result["image_list"] = retrieve_images
            state.result["text_list"] = retrieve_texts

        if state.result["image_list"]:
            img_list = [i[1] for i in state.result["image_list"]]
            logits = [i[0] for i in state.result["image_list"]]
            _names = [os.path.basename(i) for i in img_list]
            if state["show_score_key"] == "显示":
                _names = [f"{n}({i:.3f})" for i, n in zip(logits, _names)]

            with tab0_ph.container():
                image_select(
                    f"检索数量：{len(img_list)}",
                    img_list,
                    _names,
                    use_container_width=False,
                )
        else:
            tab0_ph.warning("没有检索结果", icon="🚨")

        # 展示文本信息
        if state.result["text_list"]:
            tab1_ph.table(state.result["text_list"])
        else:
            tab1_ph.warning("没有检索结果", icon="🚨")
    else:
        tabs = sac.tabs(
            [
                sac.TabsItem("数据库管理", icon="sliders"),
                sac.TabsItem("数据上传", icon="file-richtext"),
            ],
            variant="outline",
            color="green",
        )

        if tabs == "数据上传":

            def _cs():
                state.change_select_upload = True

            files = st.file_uploader(
                "xx",
                type=["png", "jpg", "jpeg", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                on_change=_cs,
                key="upload_file_key",
            )

            ph = st.empty()
            btn = ph.button(":green[**👉读取并保存👈**]", use_container_width=True)
            if btn:
                if not files:
                    st.toast(":red[请选择文件]", icon="🚨")
                elif not state.change_select_upload:
                    st.toast(":red[重复保存]", icon="🚨")
                else:
                    with ph.status(
                        ":green[**正在处理中...**]", expanded=True
                    ) as status:
                        st.markdown(now_time() + "**正在保存数据...**")
                        os.makedirs("database/files/", exist_ok=True)

                        text_files, image_files = [], []
                        for file in files:
                            if file.name.endswith(("jpg", "png", "jpeg")):
                                path = f"database/images/{file.name}"
                                with open(path, "wb") as f:
                                    f.write(file.getvalue())
                                image_files.append(path)
                            else:
                                path = f"database/files/{file.name}"
                                with open(path, "wb") as f:
                                    f.write(file.getvalue())
                                text_files.append(path)

                        if text_files:
                            st.markdown(now_time() + "**正在提取文本特征...**")
                            payload = dict(
                                text_files=text_files,
                                batch_size=BATCH_SIZE,
                                chunk_size=CHUNK_SIZE,
                                recreate_collection=False,
                            )
                            requests.post(BASE_URL + "/create_collection", json=payload)
                        if image_files:
                            st.markdown(now_time() + "**正在提取图片特征...**")
                            payload = dict(
                                image_files=image_files,
                                batch_size=BATCH_SIZE,
                                chunk_size=CHUNK_SIZE,
                                recreate_collection=False,
                            )
                            requests.post(BASE_URL + "/create_collection", json=payload)
                        cache_data.clear()
                        _ = cache_data()
                        state.change_select_upload = False
                        state.result["image_list"] = []
                        state.result["text_list"] = []
                        status.update(
                            label="📣📣:orange[**处理完毕.**]",
                            state="complete",
                            expanded=False,
                        )
        else:
            DATA_HEIGHT = 460
            data_ph = st.container(height=DATA_HEIGHT, border=False).empty()
            cols = st.columns([0.2, 0.4, 0.4])
            btn0 = cols[0].button(":green[刷新]", use_container_width=True)
            btn1 = cols[1].button(":red[删除所选]", use_container_width=True)
            btn2 = cols[2].button(
                ":red[**清空数据库**]",
                use_container_width=True,
            )

            with data_ph.container(), st.spinner("正在加载数据中...", show_time=True):
                data = cache_data()
            edited_data = data_ph.data_editor(
                data, use_container_width=True, hide_index=True, height=DATA_HEIGHT - 20
            )
            select_d = [
                [i, t, c]
                for i, t, c, s in zip(
                    edited_data["id"],
                    edited_data["type"],
                    edited_data["content"],
                    edited_data["select"],
                )
                if s
            ]

            if btn0:
                cache_data.clear()
                st.rerun()
            if btn1:
                with data_ph.container(), st.spinner("正在处理中...", show_time=True):
                    _tmp = {}
                    for d in select_d:
                        if d[1] not in _tmp:
                            _tmp[d[1]] = [d[0]]
                        else:
                            _tmp[d[1]].append(d[0])

                        # 删除对应的图片
                        if d[1] == "image":
                            if os.path.exists(d[2]):
                                os.remove(d[2])
                    # 删除vector
                    for k, v in _tmp.items():
                        payload = dict(collection_name=k, point_ids=v)
                        requests.post(BASE_URL + "/del_collection_points", json=payload)
                cache_data.clear()
                state.result["image_list"] = []
                state.result["text_list"] = []
                st.rerun()

            if btn2:
                with data_ph.container(), st.spinner("正在处理中...", show_time=True):
                    _tmp = {}
                    for d in zip(data["id"], data["type"], data["content"]):
                        if d[1] not in _tmp:
                            _tmp[d[1]] = [d[0]]
                        else:
                            _tmp[d[1]].append(d[0])

                        # 删除对应的图片
                        if d[1] == "image":
                            if os.path.exists(d[2]):
                                os.remove(d[2])

                    # 删除vector
                    for k, v in _tmp.items():
                        payload = dict(collection_name=k, point_ids=v)
                        requests.post(BASE_URL + "/del_collection_points", json=payload)

                cache_data.clear()
                state.result["image_list"] = []
                state.result["text_list"] = []
                st.rerun()


def main():
    # login_gate()
    init()
    sidebar()
    body()


if __name__ == "__main__":
    main()
