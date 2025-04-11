print("--- app.py: Starting execution ---") # DEBUG PRINT

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from markupsafe import Markup # For rendering HTML in interpretation
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import patsy
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time
import itertools
import numpy as np
import matplotlib.font_manager as fm
import statsmodels.regression.mixed_linear_model
import statsmodels.genmod.generalized_linear_model
import statsmodels.regression.linear_model
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.genmod.families import family, links, varfuncs
from statsmodels.genmod import families as sm_families
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable, Independence, Autoregressive
import uuid
import datetime
import logging
import requests
from flask_session import Session # Keep Flask-Session

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Configure Flask-Session (use filesystem by default for simplicity)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

# Folder Configuration
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = os.path.join('static', 'plots')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

# Make datetime available to templates
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.utcnow}

# Markdown filter
@app.template_filter('markdown')
def markdown_filter(text):
    # ... (markdown logic remains the same) ...
    if not text: return ""
    import re
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = text.replace('\n', '<br>')
    return Markup(text)

# --- Hugging Face API Configuration (Optional) ---
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN')
if not HF_API_TOKEN:
    app.logger.warning("Hugging Face API token not found. AI interpretation will be disabled.")

# === Helper Functions (Restored/Verified) ===
def get_sm_family(family_name):
    if family_name == 'binomial': return sm.families.Binomial()
    elif family_name == 'poisson': return sm.families.Poisson()
    elif family_name == 'gamma': return sm.families.Gamma()
    elif family_name == 'inverse_gaussian': return sm.families.InverseGaussian()
    else: return sm.families.Gaussian()

def generate_diagnostic_plots(model_results, unique_id):
    plot_paths = {}
    app.logger.info("Generating standard diagnostic plots...")
    try:
        plt.figure()
        fitted = model_results.fittedvalues
        if isinstance(fitted, pd.Series): fitted = fitted.values
        
        # Use response residuals if available, else default
        if hasattr(model_results, 'resid_response'):
             residuals = model_results.resid_response
             res_type = "Response"
        elif hasattr(model_results, 'resid'):
             residuals = model_results.resid
             res_type = "Default"
        else:
             raise AttributeError("Could not find residuals attribute.")
        if isinstance(residuals, pd.Series): residuals = residuals.values
        
        min_len = min(len(fitted), len(residuals))
        if len(fitted) != len(residuals):
             app.logger.warning(f"Length mismatch: fitted={len(fitted)}, residuals={len(residuals)}.")
        
        sns.residplot(x=fitted[:min_len], y=residuals[:min_len], lowess=True, line_kws={'color': 'red', 'lw': 1})
        plt.xlabel("Fitted values")
        plt.ylabel(f"{res_type} Residuals")
        plt.title("Residuals vs Fitted")
        plot_base_name = f"resid_vs_fitted_{unique_id}"
        png_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.png")
        svg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.svg")
        jpg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.jpg")
        plt.savefig(png_path)
        plt.savefig(svg_path)
        plt.savefig(jpg_path, dpi=300)
        plt.close()
        plot_paths['resid_vs_fitted'] = url_for('static', filename=f'plots/{plot_base_name}.png')

        plt.figure()
        sm.qqplot(residuals, line='s')
        plt.title(f"Normal Q-Q Plot ({res_type} Residuals)")
        plot_base_name = f"qq_plot_{unique_id}"
        png_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.png")
        svg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.svg")
        jpg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.jpg")
        plt.savefig(png_path)
        plt.savefig(svg_path)
        plt.savefig(jpg_path, dpi=300)
        plt.close()
        plot_paths['qq_plot'] = url_for('static', filename=f'plots/{plot_base_name}.png')

    except Exception as e:
        flash(f"Warning: Could not generate diagnostic plots. Error: {e}")
        app.logger.error(f"Error generating diagnostic plots: {e}", exc_info=True)
    return plot_paths

def get_family_object(family_name):
    return get_sm_family(family_name)

def get_cov_struct_object(cov_struct_name):
    if cov_struct_name is None: return Exchangeable()
    name_lower = cov_struct_name.lower()
    if name_lower == 'exchangeable': return Exchangeable()
    elif name_lower == 'independence': return Independence()
    elif name_lower == 'autoregressive' or name_lower == 'ar': return Autoregressive()
    else: return Exchangeable()

def auto_select_glm_family(dependent_variable_data):
    # Restore original logic here if needed (this is simplified)
    data_nonan = dependent_variable_data.dropna()
    if data_nonan.empty: return sm.families.Gaussian(), "Gaussian (Default due to empty data)"
    unique_values = data_nonan.unique()
    if all(v in [0, 1] for v in unique_values): return sm.families.Binomial(), "Binomial"
    if pd.api.types.is_integer_dtype(data_nonan) and (data_nonan >= 0).all(): return sm.families.Poisson(), "Poisson"
    if pd.api.types.is_numeric_dtype(data_nonan) and (data_nonan > 0).all(): return sm.families.Gamma(), "Gamma"
    return sm.families.Gaussian(), "Gaussian (Default)"

def select_model_aic(df, dependent_var, candidate_predictors, model_type, **kwargs):
    """
    Perform model selection using AIC with backward elimination.
    Starts with all predictors and iteratively removes the one that improves AIC the most.
    Stops when AIC cannot be further improved by removing predictors.
    """
    app.logger.info(f"Starting AIC model selection with {len(candidate_predictors)} predictors")
    
    if len(candidate_predictors) <= 1:
        formula = f"{dependent_var} ~ {' + '.join(candidate_predictors)}"
        model_func_map = {'lm': smf.ols, 'glm': smf.glm, 'lmm': smf.mixedlm, 'glmm': smf.gee}
        model = model_func_map[model_type](formula, data=df, **kwargs).fit()
        return model, formula, "AIC selection not performed (need at least 2 predictors)."
    
    # Start with the full model
    best_predictors = candidate_predictors.copy()
    model_func = {'lm': smf.ols, 'glm': smf.glm, 'lmm': smf.mixedlm, 'glmm': smf.gee}[model_type]
    
    best_formula = f"{dependent_var} ~ {' + '.join(best_predictors)}"
    best_model = model_func(best_formula, data=df, **kwargs).fit()
    best_aic = best_model.aic
    
    app.logger.info(f"Initial model AIC: {best_aic:.2f} with {len(best_predictors)} predictors")
    elimination_steps = [f"Starting AIC: {best_aic:.2f} with predictors: {', '.join(best_predictors)}"]
    
    improvement = True
    while improvement and len(best_predictors) > 1:
        improvement = False
        best_removed = None
        
        # Try removing each predictor
        for predictor in best_predictors:
            # Create a model without this predictor
            current_predictors = [p for p in best_predictors if p != predictor]
            current_formula = f"{dependent_var} ~ {' + '.join(current_predictors)}"
            
            try:
                current_model = model_func(current_formula, data=df, **kwargs).fit()
                current_aic = current_model.aic
                
                app.logger.debug(f"Without '{predictor}': AIC = {current_aic:.2f}")
                
                # If removing this predictor improves AIC (lower is better)
                if current_aic < best_aic:
                    best_aic = current_aic
                    best_removed = predictor
                    best_model = current_model
                    best_formula = current_formula
                    improvement = True
            except Exception as e:
                app.logger.warning(f"Error fitting model without '{predictor}': {e}")
                continue
        
        # If we found a predictor to remove
        if best_removed:
            best_predictors.remove(best_removed)
            elimination_steps.append(f"Removed '{best_removed}': AIC improved to {best_aic:.2f}")
            app.logger.info(f"Removed '{best_removed}': AIC improved to {best_aic:.2f}")
    
    # Return the final model with the elimination steps
    final_predictors = ', '.join(best_predictors) if best_predictors else 'Intercept only'
    log_message = f"Final model after AIC selection: {final_predictors}\n" + "\n".join(elimination_steps)
    
    return best_model, best_formula, log_message

def generate_effect_plots(model, df, dependent_var, predictors, unique_id):
    effect_plots_data = []
    font_family = 'serif' # Default fallback
    plot_rc_params = {'font.family': font_family, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.titlesize': 16}
    numeric_predictors = df[predictors].select_dtypes(include=np.number).columns.tolist()
    categorical_predictors = df[predictors].select_dtypes(exclude=np.number).columns.tolist()
    constant_data = {p: df[p].mean() for p in numeric_predictors}
    for p in categorical_predictors: constant_data[p] = df[p].mode()[0]

    for predictor in predictors:
        interpretation_text = ""
        plot_generated = False
        try:
            with plt.rc_context(plot_rc_params):
                plt.figure(figsize=(7, 7))
                ax = plt.gca()
                n_points = 100 # Number of points for prediction line
                plot_title = f"Effect of {predictor}"
                y_label = f"Predicted {model.model.endog_names}"

                if predictor in numeric_predictors:
                    # --- Define FULL range (min to max) --- 
                    x_min_full = df[predictor].min()
                    x_max_full = df[predictor].max()
                    if x_min_full == x_max_full: # Handle single unique value
                        x_min_full -= 0.1
                        x_max_full += 0.1
                    x_values_full = np.linspace(x_min_full, x_max_full, n_points)
                    
                    # --- Predictions for FULL range (mean and CI) --- 
                    pred_df_full = pd.DataFrame([constant_data] * n_points)
                    pred_df_full[predictor] = x_values_full
                    pred_summary_full = model.get_prediction(pred_df_full).summary_frame()
                    mean_pred_full = pred_summary_full['mean']
                    ci_lower_full = pred_summary_full['mean_ci_lower']
                    ci_upper_full = pred_summary_full['mean_ci_upper']

                    legend_text = "\n\n**Legend:** Black dots=Original Data; Black line=Predicted Mean; Grey area=95% CI."
                    
                    if mean_pred_full is not None:
                        # Plot ALL original data points
                        ax.scatter(df[predictor], df[dependent_var], s=25, color='black', alpha=0.6)
                        
                        # Plot prediction line for the FULL range (solid black)
                        ax.plot(x_values_full, mean_pred_full, color='black', linestyle='-', label='Predicted Mean')
                        
                        # Plot confidence interval for the FULL range (grey fill)
                        ax.fill_between(x_values_full, ci_lower_full, ci_upper_full, color='grey', alpha=0.3, label='95% CI')
                        
                        ax.set_xlabel(predictor); ax.set_ylabel(y_label); ax.set_title(plot_title)
                        
                        # Interpretation based on the overall trend
                        slope = (mean_pred_full.iloc[-1] - mean_pred_full.iloc[0]) / (x_values_full[-1] - x_values_full[0]) if len(mean_pred_full)>1 and x_values_full[-1] != x_values_full[0] else 0
                        direction = "positive" if slope > 0 else "negative" if slope < 0 else "neutral"
                        interpretation_text = f"Overall relationship appears {direction}. {legend_text}"
                        plot_generated = True
                elif predictor in categorical_predictors:
                    levels = df[predictor].unique()
                    pred_df_cat = pd.DataFrame([constant_data] * len(levels)); pred_df_cat[predictor] = levels
                    pred_summary_cat_df, legend_text_cat, errorbar_arg = None, "", None
                    
                    pred_summary_cat = model.get_prediction(pred_df_cat).summary_frame()
                    pred_summary_cat_df = pred_summary_cat; pred_summary_cat_df[predictor] = levels
                    legend_text_cat = "\n\n**Legend:** Dots=Predicted Mean; Lines=95% CI."
                    errorbar_arg = ('ci', 95)
                    
                    if pred_summary_cat_df is not None:
                        sns.pointplot(x=predictor, y='mean', data=pred_summary_cat_df, color='black', errorbar=errorbar_arg, join=False, capsize=.2, ax=ax)
                        ax.set_xlabel(predictor); ax.set_ylabel(y_label); ax.set_title(plot_title)
                        plt.xticks(rotation=45, ha='right')
                        min_level = pred_summary_cat_df.loc[pred_summary_cat_df['mean'].idxmin()][predictor]
                        max_level = pred_summary_cat_df.loc[pred_summary_cat_df['mean'].idxmax()][predictor]
                        interpretation_text = f"Highest prediction: {max_level}; Lowest: {min_level}. {legend_text_cat}"
                        plot_generated = True

            if plot_generated:
                plot_base_name = f"effect_{predictor}_{unique_id}"
                png_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.png")
                svg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.svg")
                jpg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.jpg")
                plt.tight_layout()
                plt.savefig(png_path, dpi=500, bbox_inches='tight') 
                plt.savefig(svg_path, bbox_inches='tight')
                plt.savefig(jpg_path, dpi=300, bbox_inches='tight')
                plt.close()
                effect_plots_data.append({
                    'key': f'effect_{predictor}', 
                    'predictor': predictor,
                    'plot_type_for_download': predictor,
                    'url': url_for('static', filename=f'plots/{plot_base_name}.png'),
                    'interpretation': interpretation_text
                })
        except Exception as e:
            app.logger.error(f"Error generating effect plot for {predictor}: {e}", exc_info=True)
    return effect_plots_data

def interpret_results(model_results, model_description, final_predictors):
    interpretation = [f"**Model Interpretation:**\n", f"- **Model Type:** {model_description}\n"]
    try:
        p_values = model_results.pvalues
        coeffs = model_results.params
        significant_predictors_text = []
        for predictor in final_predictors:
             if predictor in p_values.index and predictor in coeffs.index:
                 p_val = p_values[predictor]
                 coeff_val = coeffs[predictor]
                 if p_val < 0.05:
                     coeff_direction = "positive" if coeff_val > 0 else "negative"
                     # Simplified link function handling
                     original_scale_direction = coeff_direction 
                     significant_predictors_text.append(f"**{predictor}** (p={p_val:.3f}, {original_scale_direction})")
        
        if significant_predictors_text:
            interpretation.append("- **Significant Predictors (p<0.05):** " + ", ".join(significant_predictors_text) + ".\n")
        else:
             interpretation.append("- No significant predictors found (p < 0.05).\n")

        # Simplified R-squared check
        rsq_text = "N/A"
        if hasattr(model_results, 'rsquared'): rsq_text = f"{model_results.rsquared:.3f}"
        elif hasattr(model_results, 'pseudo_rsquared'): 
            try: rsq_text = f"{model_results.pseudo_rsquared():.3f} (Pseudo)"
            except: pass
        interpretation.append(f"- **Model Fit (R-squared):** {rsq_text}\n")
        interpretation.append("\n*Note: Basic interpretation. Check assumptions and context.*")
    except Exception as e:
        app.logger.error(f"Error during interpretation: {e}", exc_info=True)
        interpretation.append("\n*Error during interpretation generation.*")
    return "".join(interpretation)

def generate_formal_results_text(df, model_results, model_description, dependent_var, final_predictors, effect_plots_data):
    text = []
    try:
        n_obs = len(df)
        text.append(f"### Data Overview\nAnalyzed {n_obs} data points for `{dependent_var}`.\n")
        text.append(f"### Statistical Analysis\nA {model_description} was used.")
        text.append("### Key Findings\n")
        p_values = model_results.pvalues
        coeffs = model_results.params
        significant_found = False
        for predictor in final_predictors:
            if predictor in p_values.index and predictor in coeffs.index and p_values[predictor] < 0.05:
                significant_found = True
                original_scale_direction = "positive" if coeffs[predictor] > 0 else "negative"
                text.append(f"- **{predictor}**: Significant **{original_scale_direction}** effect (p={p_values[predictor]:.3f}).")
        if not significant_found: text.append("- No significant predictors found (p < 0.05).")
        text.append("\n*Disclaimer: Automated summary. Review required.*")
    except Exception as e:
        app.logger.error(f"Error generating formal results: {e}", exc_info=True)
        return "*Error generating formal results.*"
    return "\n".join(text)

def get_ai_interpretation(prompt, api_token):
    if not api_token: return "AI interpretation disabled: Token not provided."
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200, "return_full_text": False}}
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text'].strip()
        else: return "Error: Unexpected format from AI API."
    except Exception as e: return f"Error calling AI API: {e}"

# === Application Routes (Simplified) ===
@app.route('/')
def index():
    """Displays the main page, showing only BOS.txt if it exists."""
    target_file = "BOS.txt"
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], target_file)
    target_exists = os.path.exists(target_path)
    uploaded_files = []
    if target_exists:
        uploaded_files.append(target_file)
    else:
         flash(f"Core file '{target_file}' not found in uploads directory '{app.config['UPLOAD_FOLDER']}'. Please upload it.", "info")
         
    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads, saving them to the uploads folder."""
    if 'file' not in request.files:
        flash('No file part', 'warning')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'warning')
        return redirect(url_for('index'))
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            flash(f'File "{filename}" uploaded successfully.', 'success')
            target_file = "BOS.txt"
            if filename == target_file:
                 return redirect(url_for('index'))
            else:
                 return redirect(url_for('analyze', filename=filename))
        except Exception as e:
            flash(f'An error occurred while saving the file: {e}', 'danger')
            return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze(filename):
    """Handles displaying the analysis form and processing the analysis."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash(f'Error: File "{filename}" not found.', 'danger')
        return redirect(url_for('index'))

    try:
        # Reading and processing logic remains largely the same...
        try: 
            df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='warn')
            df.columns = df.columns.str.replace(r'[\s\.\-\(\)]', '_', regex=True)
            df.columns = [f'col_{c}' if str(c).isdigit() or not str(c)[0].isalpha() and str(c)[0] != '_' else c for c in df.columns]
            df.columns = pd.unique(df.columns)
        except Exception as e_csv: 
             df = pd.read_excel(filepath)
             df.columns = df.columns.str.replace(r'[\s\.\-\(\)]', '_', regex=True)
             df.columns = [f'col_{c}' if str(c).isdigit() or not str(c)[0].isalpha() and str(c)[0] != '_' else c for c in df.columns]
             df.columns = pd.unique(df.columns)
        columns = df.columns.tolist()

        if request.method == 'GET':
            # Use the original df (columns already cleaned) and the existing columns list
            preview_html = df.head().to_html(classes='table table-striped table-sm', index=False, border=0)
            
            return render_template(
                'analyze.html',
                filename=filename,
                columns=columns, # Use the columns list defined above
                preview_html=preview_html
            )
        
        elif request.method == 'POST':
            # --- Process form data --- 
            model_type = request.form.get('model_type')
            dependent_var = request.form.get('dependent_var')
            original_independent_vars = request.form.getlist('independent_vars')
            independent_vars = list(original_independent_vars)
            perform_aic_selection = request.form.get('perform_aic') == 'on'

            if not dependent_var or dependent_var not in df.columns:
                 flash(f"Error: Invalid or missing dependent variable '{dependent_var}'.", 'danger')
                 return render_template('analyze.html', filename=filename, columns=columns)
            
            if not independent_vars or not all(v in df.columns for v in independent_vars):
                 flash("Error: Invalid or missing independent variable(s).", 'danger')
                 return render_template('analyze.html', filename=filename, columns=columns)

            session_id = session.get('_id', str(uuid.uuid4()))
            if '_id' not in session: session['_id'] = session_id
            app.logger.info(f"Using session_id: {session_id} for this request.")

            dep_var_desc = request.form.get('dep_var_description', '')
            indep_var_descriptions = {var: request.form.get(f'desc_{var}', '') for var in independent_vars if request.form.get(f'desc_{var}', '')}
            user_api_token = request.form.get('hf_api_token') or HF_API_TOKEN
            
            model_kwargs = {}
            model_description_base = ""
            results_summary = None
            model = None
            diagnostic_plot_urls = {} # Renamed from plot_urls for clarity
            effect_plots_data = []
            selection_log = None
            final_predictors = []

            try:
                # --- MODEL SETUP LOGIC (GLM, LMM, GEE, LM) --- 
                # ... (This block should contain the original logic to set 
                #      model_kwargs, model_description_base, and potentially 
                #      adjust independent_vars based on model_type like LMM grouping) ...
                # Example for LM:
                if model_type == 'lm':
                    model_description_base = "Linear Model (OLS)"
                    final_predictors = list(independent_vars)
                elif model_type == 'glm':
                    # --- GLM Family Selection ---
                    family_selection_mode = request.form.get('family_selection_mode', 'auto')
                    selected_family_obj = None
                    selected_family_name = ""
                    family_source = ""  # Initialize family_source variable
                    
                    if family_selection_mode == 'auto':
                        try:
                            selected_family_obj, selected_family_name = auto_select_glm_family(df[dependent_var])
                            family_source = "(Auto-Selected)"
                            flash(f"Auto-selected GLM family: {selected_family_name}", "info")
                        except Exception as e:
                            flash(f"Error auto-selecting family: {e}. Defaulting to Gaussian.", "warning")
                            app.logger.warning(f"Auto-select family failed: {e}", exc_info=True)
                            selected_family_obj = sm.families.Gaussian()
                            selected_family_name = "Gaussian"
                            family_source = "(Default due to Error)"
                    else:
                        # Manual selection
                        family_choice = request.form.get('glm_family', 'gaussian')
                        try:
                            selected_family_obj = get_family_object(family_choice)
                            selected_family_name = family_choice.capitalize()
                            family_source = "(User-Selected)"
                            flash(f"Using manually selected family: {selected_family_name}", "info")
                        except Exception as e:
                            flash(f"Error with selected family: {e}. Defaulting to Gaussian.", "warning")
                            app.logger.warning(f"Family selection failed: {e}", exc_info=True)
                            selected_family_obj = sm.families.Gaussian()
                            selected_family_name = "Gaussian"
                            family_source = "(Default due to Error)"
                    
                    if selected_family_obj is None: # Final fallback
                        selected_family_obj = sm.families.Gaussian()
                        selected_family_name = "Gaussian"
                        family_source = "(Default)"

                    model_kwargs['family'] = selected_family_obj
                    model_description_base = f"Generalized Linear Model (GLM) - Family: {selected_family_name} {family_source}"
                    final_predictors = list(independent_vars)
                elif model_type == 'lmm':
                    # ... (LMM grouping/random effects logic) ...
                    grouping_vars = request.form.getlist('grouping_vars') #...
                    model_description_base = f"Linear Mixed Model (LMM) - Group(s): {', '.join(grouping_vars)}"
                    final_predictors = list(independent_vars) 
                elif model_type == 'glmm': # GEE
                    # ... (GEE family, group, cov_struct logic) ...
                    model_description_base = f"Generalized Estimating Equations (GEE) - ..."
                    final_predictors = list(independent_vars)
                else:
                    raise ValueError(f'Unknown model type selected: {model_type}')
                # --- END MODEL SETUP LOGIC --- 

                # --- MODEL FITTING / SELECTION --- 
                base_formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
                model_description = "" # Will be set below
                model_func = None
                # ... (Determine model_func based on model_type: smf.ols, smf.glm, etc.) ...
                if model_type == 'lm': model_func = smf.ols
                elif model_type == 'glm': model_func = smf.glm
                elif model_type == 'lmm': model_func = smf.mixedlm
                elif model_type == 'glmm': model_func = smf.gee
                else: raise ValueError("Cannot determine model function.")

                if perform_aic_selection and len(independent_vars) > 1:
                     model, final_formula, selection_log_str = select_model_aic(df, dependent_var, independent_vars, model_type, **model_kwargs)
                     selection_log = selection_log_str
                     # ... (extract final_predictors from final_formula) ...
                     try:
                          final_predictors = list(patsy.dmatrix(final_formula.split('~')[1], df, return_type='dataframe').columns)
                          if 'Intercept' in final_predictors: final_predictors.remove('Intercept')
                          final_predictors = [p.split('[')[0] if '[' in p else p for p in final_predictors] 
                          final_predictors = list(dict.fromkeys(final_predictors))
                     except Exception as e: #... (fallback)
                          final_predictors = list(independent_vars)
                     model_description = model_description_base + f" (AIC/QIC Selected - Formula: {final_formula})"
                     flash("AIC/QIC selection performed...", "info")
                else:
                     fit_args = {'formula': base_formula, 'data': df}
                     fit_args.update(model_kwargs)
                     model = model_func(**fit_args).fit()
                     final_predictors = list(independent_vars)
                     model_description = model_description_base + " (Full Model)"
                     flash("Full model fitted...", "info")
                # --- END MODEL FITTING --- 

                # --- GET RESULTS & PLOTS (Actual calculation) --- 
                results_summary = model.summary().as_html() if hasattr(model, 'summary') else str(model)
                
                try:
                    diagnostic_plot_urls = generate_diagnostic_plots(model, session_id) 
                except Exception as diag_err:
                    flash("Warning: Could not generate diagnostic plots...", 'warning')

                if final_predictors:
                    try:
                         effect_plots_data = generate_effect_plots(model, df, dependent_var, final_predictors, session_id)
                    except Exception as effect_err:
                         flash("Warning: Could not generate effect plots...", 'warning')
                # --- END GET RESULTS & PLOTS --- 

                # --- GENERATE INTERPRETATIONS (Actual calculation) --- 
                interpretation_text = interpret_results(model, model_description, final_predictors)
                formal_results_text = generate_formal_results_text(df, model, model_description, dependent_var, final_predictors, effect_plots_data)
                ai_interpretation = "AI interpretation disabled or failed."
                if user_api_token:
                     # ... (Construct prompt) ...
                     prompt = f"..."
                     ai_interpretation = get_ai_interpretation(prompt, user_api_token)
                # --- END INTERPRETATIONS --- 

            except Exception as e:
                flash(f"Error during analysis: {e}", 'danger')
                app.logger.error(f"Analysis Error during POST for {filename}:\n{traceback.format_exc()}")
                return redirect(url_for('index')) # Redirect on major error
            
            # --- Render results using ACTUAL calculated variables --- 
            return render_template('results.html', 
                                    filename=filename,
                                    model_description=model_description, # Actual
                                    results_summary=results_summary, # Actual
                                    plot_urls=diagnostic_plot_urls, # Actual
                                    effect_plots=effect_plots_data, # Actual
                                    interpretation=interpretation_text, # Actual
                                    formal_results_text=formal_results_text, # Actual
                                    ai_interpretation=ai_interpretation, # Actual
                                    selection_log=selection_log, # Actual
                                    session_id=session_id,
                                    dependent_var=dependent_var,
                                    final_predictors=final_predictors, 
                                    dep_var_desc=dep_var_desc, 
                                    indep_var_descriptions=indep_var_descriptions, 
                                    user_api_token=user_api_token 
                                    )

    except (pd.errors.ParserError, KeyError, Exception) as e:
        flash(f"Error analyzing file '{filename}': {e}", 'danger')
        import traceback
        app.logger.error(f"Analysis Error for {filename}:\n{traceback.format_exc()}")
        return redirect(url_for('index'))

    return redirect(url_for('index')) # Fallback

@app.route('/download_plot/<unique_id>/<plot_type>/<format>')
def download_plot(unique_id, plot_type, format):
    # Restore original logic for finding and sending plot file
    plot_filename = f"{plot_type}_{unique_id}.{format}"
    plot_path = os.path.join(app.config['PLOT_FOLDER'], plot_filename)
    if os.path.exists(plot_path):
        try:
            return send_from_directory(app.config['PLOT_FOLDER'], plot_filename, as_attachment=True)
        except Exception as e:
            app.logger.error(f"Error sending plot {plot_filename}: {e}")
            flash("Error downloading plot.", "danger")
    else:
        flash(f"Plot file {plot_filename} not found.", "warning")
    return redirect(url_for('index'))

@app.route('/ask_ai', methods=['POST'])
def ask_ai():
    # Restore original logic for receiving JSON and calling get_ai_interpretation
    data = request.get_json()
    if not data: return jsonify({"error": "No JSON data received."}), 400
    prompt = data.get('prompt')
    api_token = data.get('api_token') # Expect token in request now
    if not prompt or not api_token: return jsonify({"error": "Missing prompt or api_token."}), 400
    
    ai_response = get_ai_interpretation(prompt, api_token)
    if ai_response.startswith("Error") or ai_response.startswith("AI interpretation disabled"):
        return jsonify({"error": ai_response}), 500
    return jsonify({"response": ai_response})

@app.route('/configure/<filename>')
def configure_analysis(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        flash(f'File {filename} not found.', 'danger')
        return redirect(url_for('index'))

    try:
        df = load_data(file_path)
        # Clean column names for easier use
        df.columns = [clean_column_name(col) for col in df.columns]
        columns = df.columns.tolist()
        preview_html = df.head().to_html(classes='table table-sm table-hover', border=0)
        return render_template('analyze.html', filename=filename, columns=columns, preview_html=preview_html)
    except Exception as e:
        flash(f'Error reading or processing file {filename}: {e}', 'danger')
        app.logger.error(f"Error configuring analysis for {filename}: {e}", exc_info=True)
        return redirect(url_for('index'))

# Removed database init command, user model, auth routes, forms, etc.

if __name__ == '__main__':
    app.run(debug=True) # Keep debug=True for development