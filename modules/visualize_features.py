
import numpy as np
import plotly.express as px
from plotly.express import scatter_3d


def visualize_features(
    features: np.ndarray, 
    labels: list[str],
    classes: list[str], 
    save_path: str = "feature_mapping.html"
) -> None:
    """特徴量を3Dで可視化する関数
    
    Args:
        features (np.ndarray): 抽出した特徴量 (num_samples, 3)
        labels (list[str]): 各サンプルのクラスラベル
        classes (list[str]): クラス名のリスト
        save_path (str, optional): 可視化結果の保存パス
    """
    # ラベル数分の色を作成
    color_map = px.colors.sample_colorscale(
        px.colors.sequential.Turbo,
        np.linspace(0, 1, len(classes))
    )
    
    # 3次元散布図の作成
    fig = scatter_3d(
        x=features[:, 0],
        y=features[:, 1],
        z=features[:, 2],
        color=[f"{classes.index(classes[label])}-{classes[label]}" for label in labels],
        color_discrete_sequence=color_map,
        labels={"color": "Classes"},
    )
    # 設定の調整
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[features[:,0].min(), features[:,0].max()], autorange=False),
            yaxis=dict(range=[features[:,1].min(), features[:,1].max()], autorange=False),
            zaxis=dict(range=[features[:,2].min(), features[:,2].max()], autorange=False),
            camera=dict(
                eye=dict(x=1.25, y=1.25, z=1.25),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            )
        ),
        uirevision="fixed"
    )
    
    # 表示
    fig.show()
    # HTMLファイルとして保存
    fig.write_html(save_path)
