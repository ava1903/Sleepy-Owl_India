@st.cache_data
def train_regression(feats_tuple):
    from sklearn.linear_model import Ridge, Lasso
    feats = list(feats_tuple)
    df = load_data()
    df2 = df.dropna(subset=["q14_monthly_spend_inr"])
    X = df2[feats].fillna(df2[feats].median())
    y = df2["q14_monthly_spend_inr"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    lr    = LinearRegression()
    ridge = Ridge(alpha=10.0)
    lasso = Lasso(alpha=5.0, max_iter=5000)
    rf    = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)

    lr.fit(Xtr_s, ytr);    ridge.fit(Xtr_s, ytr)
    lasso.fit(Xtr_s, ytr); rf.fit(Xtr, ytr)

    yp_lr    = lr.predict(Xte_s)
    yp_ridge = ridge.predict(Xte_s)
    yp_lasso = lasso.predict(Xte_s)
    yp_rf    = rf.predict(Xte)

    def _m(yp):
        return {
            "r2":   round(float(r2_score(yte, yp)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(yte, yp))), 1),
            "mae":  round(float(mean_absolute_error(yte, yp)), 1),
            "mse":  round(float(mean_squared_error(yte, yp)), 1),
        }

    return {
        "lr": lr, "ridge": ridge, "lasso": lasso, "rf": rf, "sc": sc,
        "Xtr": pd.DataFrame(Xtr, columns=feats),
        "Xte": Xte, "ytr": ytr, "yte": yte,
        "yp_lr": yp_lr, "yp_ridge": yp_ridge, "yp_lasso": yp_lasso, "yp_rf": yp_rf,
        "metrics_lr":    _m(yp_lr),
        "metrics_ridge": _m(yp_ridge),
        "metrics_lasso": _m(yp_lasso),
        "metrics_rf":    _m(yp_rf),
        "coefs_lr":    pd.Series(lr.coef_,    index=feats),
        "coefs_ridge": pd.Series(ridge.coef_, index=feats),
        "coefs_lasso": pd.Series(lasso.coef_, index=feats),
        "fi_rf":       pd.Series(rf.feature_importances_, index=feats),
        "feats": feats,
    }


def _reg_tab(tab, mname, yp, yte, mets, coefs, is_lasso=False):
    with tab:
        st.markdown(f"*Model: **{mname}***")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("R2 Score", str(mets["r2"]), mname), unsafe_allow_html=True)
        c2.markdown(metric_card("RMSE",  f"Rs.{mets['rmse']:,.1f}", "Root Mean Sq Error"), unsafe_allow_html=True)
        c3.markdown(metric_card("MAE",   f"Rs.{mets['mae']:,.1f}",  "Mean Abs Error"),      unsafe_allow_html=True)
        c4.markdown(metric_card("MSE",   f"Rs.{mets['mse']:,.1f}",  "Mean Sq Error"),       unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            section_header("Actual vs Predicted")
            lim = max(float(yte.max()), float(yp.max())) * 1.08
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yte, y=yp, mode="markers",
                marker=dict(color=TEAL, opacity=0.4, size=5), name="Predictions"))
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                line=dict(color=RED, dash="dash", width=2), name="Perfect Fit"))
            fig.update_layout(xaxis_title="Actual (Rs.)", yaxis_title="Predicted (Rs.)",
                annotations=[dict(x=0.05, y=0.93, xref="paper", yref="paper",
                    text=f"R2={mets['r2']}", showarrow=False,
                    font=dict(color=ACCENT, size=13, family="monospace"))])
            dark_fig(fig, 360)
            st.plotly_chart(fig, width="stretch")
        with col2:
            section_header("Residuals Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=yte - yp, nbinsx=40,
                marker=dict(color=ACCENT, opacity=0.75), name="Residuals"))
            fig.add_vline(x=0, line=dict(color=RED, dash="dash", width=1.5))
            fig.update_layout(xaxis_title="Residual (Rs.)", yaxis_title="Count", showlegend=False)
            dark_fig(fig, 360)
            st.plotly_chart(fig, width="stretch")
        section_header(f"Coefficients - {mname} (Standardised)")
        cs = coefs.sort_values()
        show_c = cs[cs.abs() > 0.001] if is_lasso else cs
        fig = go.Figure(go.Bar(
            x=show_c.values,
            y=[REG_LABELS.get(f, f) for f in show_c.index],
            orientation="h",
            marker=dict(color=[ACCENT if v >= 0 else RED for v in show_c.values]),
            text=[f"{v:+.3f}" for v in show_c.values],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="Coefficient (Standardised)", yaxis_title="")
        dark_fig(fig, 440)
        st.plotly_chart(fig, width="stretch")
        if is_lasso:
            n_zero = int((coefs.abs() < 0.001).sum())
            st.markdown(insight_card(
                f"<strong>Lasso shrinkage:</strong> {n_zero} of {len(coefs)} features "
                "driven to zero - automatic feature selection."
            ), unsafe_allow_html=True)


def page_regression(df):
    page_title(
        "Sleepy Owl Coffee - Cross-Sell Intelligence Dashboard",
        "Regression Analysis - What drives monthly coffee spend?"
    )

    res = train_regression(tuple(REG_FEATURES))
    yte = res["yte"]

    tab_lr, tab_ridge, tab_lasso, tab_rf, tab_sim = st.tabs([
        "Linear Regression", "Ridge Regression",
        "Lasso Regression",  "Random Forest",
        "Spend Simulator",
    ])

    _reg_tab(tab_lr,    "Linear Regression", res["yp_lr"],    yte, res["metrics_lr"],    res["coefs_lr"],    False)
    _reg_tab(tab_ridge, "Ridge (a=10)",       res["yp_ridge"], yte, res["metrics_ridge"], res["coefs_ridge"], False)
    _reg_tab(tab_lasso, "Lasso (a=5)",        res["yp_lasso"], yte, res["metrics_lasso"], res["coefs_lasso"], True)

    with tab_rf:
        mets = res["metrics_rf"]
        yp   = res["yp_rf"]
        st.markdown("*Model: **Random Forest Regressor (150 trees, max_depth=8)***")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("R2 Score", str(mets["r2"]),           "Random Forest"),       unsafe_allow_html=True)
        c2.markdown(metric_card("RMSE",  f"Rs.{mets['rmse']:,.1f}",    "Root Mean Sq Error"),  unsafe_allow_html=True)
        c3.markdown(metric_card("MAE",   f"Rs.{mets['mae']:,.1f}",     "Mean Abs Error"),       unsafe_allow_html=True)
        c4.markdown(metric_card("MSE",   f"Rs.{mets['mse']:,.1f}",     "Mean Sq Error"),        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            section_header("Actual vs Predicted")
            lim = max(float(yte.max()), float(yp.max())) * 1.08
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=yte, y=yp, mode="markers",
                marker=dict(color=TEAL, opacity=0.4, size=5), name="RF Predictions"))
            fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                line=dict(color=RED, dash="dash", width=2), name="Perfect Fit"))
            fig.update_layout(xaxis_title="Actual (Rs.)", yaxis_title="Predicted (Rs.)")
            dark_fig(fig, 360)
            st.plotly_chart(fig, width="stretch")
        with col2:
            section_header("Residuals Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=yte - yp, nbinsx=40,
                marker=dict(color=HIGHLIGHT, opacity=0.75), name="Residuals"))
            fig.add_vline(x=0, line=dict(color=RED, dash="dash", width=1.5))
            fig.update_layout(xaxis_title="Residual (Rs.)", yaxis_title="Count", showlegend=False)
            dark_fig(fig, 360)
            st.plotly_chart(fig, width="stretch")
        section_header("Feature Importances - Random Forest")
        fi = res["fi_rf"].sort_values()
        fig = go.Figure(go.Bar(
            x=fi.values,
            y=[REG_LABELS.get(f, f) for f in fi.index],
            orientation="h",
            marker=dict(color=[ACCENT if v >= fi.median() else TEAL for v in fi.values]),
            text=[f"{v:.3f}" for v in fi.values],
            textposition="outside",
        ))
        fig.update_layout(xaxis_title="Importance Score", yaxis_title="")
        dark_fig(fig, 460)
        st.plotly_chart(fig, width="stretch")
        section_header("All Models - Side-by-Side Comparison")
        comp_rows = [
            {"Model": "Linear Regression", **res["metrics_lr"]},
            {"Model": "Ridge (a=10)",       **res["metrics_ridge"]},
            {"Model": "Lasso (a=5)",        **res["metrics_lasso"]},
            {"Model": "Random Forest",       **res["metrics_rf"]},
        ]
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(
            comp_df.style
                .highlight_max(subset=["r2"],               color="#D4EDDA")
                .highlight_min(subset=["rmse", "mae", "mse"], color="#D4EDDA"),
            width="stretch", hide_index=True,
        )
        fig = go.Figure()
        for metric, color, label in [("r2", ACCENT, "R2"), ("rmse", TEAL, "RMSE"), ("mae", AMBER, "MAE")]:
            fig.add_trace(go.Bar(name=label, x=comp_df["Model"], y=comp_df[metric],
                marker_color=color, text=comp_df[metric], textposition="outside"))
        fig.update_layout(barmode="group", title="Model Comparison - R2, RMSE, MAE",
                          yaxis_title="Score / Error", xaxis_title="")
        dark_fig(fig, 360)
        st.plotly_chart(fig, width="stretch")

    with tab_sim:
        section_header("Spend Prediction Simulator")
        st.markdown("*Adjust customer profile - all four models predict simultaneously.*")
        left, right = st.columns([1, 1])
        with left:
            sim_vals = {}
            for feat in REG_FEATURES:
                col_data = df[feat].dropna()
                min_v  = float(col_data.min())
                max_v  = float(col_data.max())
                mean_v = float(col_data.mean())
                label  = REG_LABELS.get(feat, feat)
                if max_v - min_v > 1:
                    sim_vals[feat] = st.slider(label, min_v, max_v, mean_v,
                        step=0.5 if max_v <= 10 else 50.0, key=f"sim_{feat}")
                else:
                    sim_vals[feat] = float(st.selectbox(label, [0, 1], index=int(round(mean_v)),
                        format_func=lambda x: "Yes" if x else "No", key=f"simb_{feat}"))
        with right:
            sim_arr    = pd.DataFrame([[sim_vals[f] for f in REG_FEATURES]], columns=REG_FEATURES)
            sim_scaled = res["sc"].transform(sim_arr)
            pred_lr    = float(res["lr"].predict(sim_scaled)[0])
            pred_ridge = float(res["ridge"].predict(sim_scaled)[0])
            pred_lasso = float(res["lasso"].predict(sim_scaled)[0])
            pred_rf    = float(res["rf"].predict(sim_arr)[0])
            pc1, pc2 = st.columns(2)
            pc3, pc4 = st.columns(2)
            pc1.markdown(metric_card("Linear Reg",   f"Rs.{pred_lr:,.0f}",    "Prediction"), unsafe_allow_html=True)
            pc2.markdown(metric_card("Ridge (a=10)", f"Rs.{pred_ridge:,.0f}", "Prediction"), unsafe_allow_html=True)
            pc3.markdown(metric_card("Lasso (a=5)",  f"Rs.{pred_lasso:,.0f}", "Prediction"), unsafe_allow_html=True)
            pc4.markdown(metric_card("Random Forest",f"Rs.{pred_rf:,.0f}",    "Best Fit"),   unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            Xtr_means = {f: float(res["Xtr"][f].mean()) for f in REG_FEATURES}
            baseline  = float(res["rf"].predict(pd.DataFrame([Xtr_means]))[0])
            contribs  = []
            for feat in REG_FEATURES:
                row = dict(Xtr_means); row[feat] = sim_vals[feat]
                delta = float(res["rf"].predict(pd.DataFrame([row]))[0]) - baseline
                contribs.append((REG_LABELS.get(feat, feat), delta))
            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_c   = contribs[:8]
            total_d = sum(c[1] for c in top_c)
            fig_wf = go.Figure(go.Waterfall(
                orientation="h",
                measure=["relative"] * len(top_c) + ["total"],
                y=[c[0] for c in top_c] + ["Net Impact"],
                x=[c[1] for c in top_c] + [total_d],
                text=[f"Rs.{c[1]:+.0f}" for c in top_c] + [f"Rs.{total_d:+.0f}"],
                textposition="outside",
                decreasing=dict(marker=dict(color=RED)),
                increasing=dict(marker=dict(color=HIGHLIGHT)),
                totals=dict(marker=dict(color=ACCENT)),
            ))
            fig_wf.update_layout(title="RF Spend Drivers vs Average Customer",
                                  xaxis_title="Rs. Impact")
            dark_fig(fig_wf, 440)
            st.plotly_chart(fig_wf, width="stretch")

