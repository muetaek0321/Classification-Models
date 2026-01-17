
import numpy as np
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
    fig = scatter_3d(
        x=features[:, 0],
        y=features[:, 1],
        z=features[:, 2],
        color=[f"{classes.index(classes[label])}-{classes[label]}" for label in labels],
        title="Feature Visualization with UMAP",
        labels={"color": "Classes"}
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(scene=dict(
        xaxis=dict(range=[features[:,0].min(), features[:,0].max()]),
        yaxis=dict(range=[features[:,1].min(), features[:,1].max()]),
        zaxis=dict(range=[features[:,2].min(), features[:,2].max()]))
    )
    
    fig.write_html(save_path)
