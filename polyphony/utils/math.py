import pandas as pd


def cluster_agg(latent_mat, cluster_assignment):
    inner_var_df = pd.DataFrame(latent_mat)
    inner_var_df['cluster'] = cluster_assignment.argmax(axis=0)
    inner_var_df = inner_var_df[cluster_assignment.sum(axis=0) > 0]
    query_cluster_group = inner_var_df.groupby('cluster')
    return query_cluster_group.agg(["mean", "var"])
