print("--- app.py: Starting execution ---") # DEBUG PRINT

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from markupsafe import Markup # For rendering HTML in interpretation
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import patsy # Needed for formula handling, especially with categorical data
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time # To generate unique filenames for plots
import itertools
import numpy as np # For numerical operations
import matplotlib.font_manager as fm # For font checks
import statsmodels.regression.mixed_linear_model
import statsmodels.genmod.generalized_linear_model
import statsmodels.regression.linear_model
from statsmodels.tools.eval_measures import aic, bic # Keep AIC/BIC for model selection
from statsmodels.genmod.families import family # For GLM/GEE families
from statsmodels.genmod.families import links # For GLM/GEE links
from statsmodels.genmod.families import varfuncs # For GLM/GEE variance functions
from statsmodels.genmod import families as sm_families # Easier access to families and cov structures
from statsmodels.genmod.generalized_estimating_equations import GEE # Import GEE class if needed directly
from statsmodels.genmod.cov_struct import ( # Import common covariance structures for GEE
    Exchangeable, Independence, Autoregressive
)
import uuid
import datetime # Add this import
import logging # Re-add the missing import
import requests # Add requests import
import sys # Add sys import
from flask_session import Session # Add this import
from flask_sqlalchemy import SQLAlchemy             # Import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user # Import Flask-Login components
from werkzeug.security import generate_password_hash, check_password_hash # For passwords
from dotenv import load_dotenv                     # For local .env file
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError

load_dotenv() # Load environment variables from .env file (if it exists)

# Initialize Flask app - it will find 'templates' and 'static' folders by default
app = Flask(__name__) # Removed template_folder and static_folder arguments

# --- Database Configuration --- 
db_url = os.environ.get('DATABASE_URL')
if db_url and db_url.startswith("postgres://"): # Fix for newer Heroku/Render URLs
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url or 'sqlite:///local_dev.db' # Use DATABASE_URL or fallback to local sqlite
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # Disable modification tracking

db = SQLAlchemy(app) # Initialize SQLAlchemy
# ---------------------------

# --- Login Manager Configuration --- 
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Redirect to 'login' view if user tries to access protected page
# ----------------------------------

app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24)) # Use env var or random
# Flask-Session config might need adjustment depending on how it integrates with SQLAlchemy/Login
# app.config['SESSION_TYPE'] = 'filesystem'
# app.config['SESSION_PERMANENT'] = False
# Session(app)

# === User Model ===
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False) # Added email
    password_hash = db.Column(db.String(256), nullable=False) # Increased length for stronger hashes

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'
# =================

# === Flask-Login User Loader ===
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id)) # Use db.session.get for SQLAlchemy 2.0+
# =============================

# Define simple relative paths for uploads and plots
# Note: Filesystem might be ephemeral on free hosting tiers!
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = os.path.join('static', 'plots')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PLOT_FOLDER'] = PLOT_FOLDER

# Ensure folders exist (will be created relative to where the app runs)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
# Removed logging of absolute paths

# Make datetime available to templates for footer year
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.utcnow}

# Helper function for rendering markdown in templates
@app.template_filter('markdown')
def markdown_filter(text):
    if not text: return ""
    import re
    # Replace **bold** with <b>bold</b> using regex (non-greedy)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace `code` with <code>code</code> using regex
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    # Replace newlines with <br>
    text = text.replace('\n', '<br>')
    return Markup(text)

# --- Hugging Face API Configuration ---
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1" # Example model
HF_API_TOKEN = os.environ.get('HF_API_TOKEN') # Read token from environment variable

if not HF_API_TOKEN:
    app.logger.warning("Hugging Face API token (HF_API_TOKEN) not found in environment variables. AI interpretation will be disabled.")

# --- Helper function to get statsmodels family object ---
def get_sm_family(family_name):
    if family_name == 'binomial':
        return sm.families.Binomial()
    elif family_name == 'poisson':
        return sm.families.Poisson()
    elif family_name == 'gamma':
        return sm.families.Gamma()
    elif family_name == 'inverse_gaussian':
        return sm.families.InverseGaussian()
    # Add other families as needed
    else: # Default to Gaussian
        return sm.families.Gaussian()

# --- Restore original diagnostic plots function (ensure uses resid_response if available) ---
def generate_diagnostic_plots(model_results, unique_id):
    """Generates diagnostic plots and saves them, returning their paths."""
    plot_paths = {}
    app.logger.info("Generating standard diagnostic plots...")
    try:
        plt.figure()
        fitted = model_results.fittedvalues
        if isinstance(fitted, pd.Series): fitted = fitted.values
        
        # Use response residuals if available (better for GLM/GLMM), otherwise use standard residuals
        if hasattr(model_results, 'resid_response'):
             residuals = model_results.resid_response
             res_type = "Response"
        elif hasattr(model_results, 'resid'):
             residuals = model_results.resid
             res_type = "Default"
        else:
             raise AttributeError("Could not find residuals attribute on model results.")
        if isinstance(residuals, pd.Series): residuals = residuals.values
        app.logger.info(f"Using {res_type} residuals for diagnostic plots.")

        app.logger.info(f"Plotting residuals vs fitted. Residuals shape: {residuals.shape}, Fitted shape: {fitted.shape}")
        min_len = min(len(fitted), len(residuals))
        if len(fitted) != len(residuals):
             app.logger.warning(f"Length mismatch: fitted={len(fitted)}, residuals={len(residuals)}. Using min_len={min_len}")
        
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
        plt.savefig(jpg_path, dpi=300) # Save JPEG with decent resolution
        plt.close()
        plot_paths['resid_vs_fitted'] = url_for('static', filename=f'plots/{plot_base_name}.png') # Keep PNG for display
        app.logger.info(f"Saved resid_vs_fitted plots (png, svg, jpg) for {unique_id}")

        plt.figure()
        sm.qqplot(residuals, line='s')
        plt.title(f"Normal Q-Q Plot ({res_type} Residuals)") # Note: Normality assumption might not apply directly depending on model
        plot_base_name = f"qq_plot_{unique_id}"
        png_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.png")
        svg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.svg")
        jpg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.jpg")
        plt.savefig(png_path)
        plt.savefig(svg_path)
        plt.savefig(jpg_path, dpi=300)
        plt.close()
        plot_paths['qq_plot'] = url_for('static', filename=f'plots/{plot_base_name}.png') # Keep PNG for display
        app.logger.info(f"Saved qq_plot plots (png, svg, jpg) for {unique_id}")

    except Exception as e:
        flash(f"Warning: Could not generate diagnostic plots. Error: {e}")
        app.logger.error(f"Error generating diagnostic plots: {e}", exc_info=True)
        os.makedirs(PLOT_FOLDER, exist_ok=True)

    return plot_paths

# --- Helper function to map family name strings to statsmodels objects ---
def get_family_object(family_name):
    if family_name == 'binomial':
        return sm.families.Binomial()
    elif family_name == 'poisson':
        return sm.families.Poisson()
    elif family_name == 'gamma':
        return sm.families.Gamma()
    elif family_name == 'inverse_gaussian':
        return sm.families.InverseGaussian()
    # Add other families as needed
    else: # Default to Gaussian
        return sm.families.Gaussian()

# --- Function to map cov struct name strings to statsmodels objects (NEW for GEE) ---
def get_cov_struct_object(cov_struct_name):
    """Maps covariance structure name string to statsmodels object."""
    if cov_struct_name is None: # Default if not specified
        return Exchangeable()
    name_lower = cov_struct_name.lower()
    if name_lower == 'exchangeable':
        return Exchangeable()
    elif name_lower == 'independence':
        return Independence()
    elif name_lower == 'autoregressive' or name_lower == 'ar':
        return Autoregressive()
    else:
        app.logger.warning(f"Unknown covariance structure '{cov_struct_name}'. Defaulting to Exchangeable.")
        return Exchangeable() # Default to Exchangeable for unknown

# --- Helper function to automatically select GLM/GEE family based on dependent variable ---
def auto_select_glm_family(dependent_variable_data):
    """Selects a plausible GLM family based on dependent variable data."""
    app.logger.info(f"Auto-selecting GLM family for variable type: {dependent_variable_data.dtype}")
    data_nonan = dependent_variable_data.dropna()
    if data_nonan.empty:
         app.logger.warning("Dependent variable has no non-NA values. Defaulting to Gaussian.")
         return sm.families.Gaussian(), "Gaussian (Default due to empty data)"
         
    unique_values = data_nonan.unique()
    
    # Check for Binary (0/1)
    if all(v in [0, 1] for v in unique_values):
         app.logger.info("Detected Binary (0/1) data. Selecting Binomial family.")
         return sm.families.Binomial(), "Binomial"
         
    # Check for Integer Count Data (>= 0)
    if pd.api.types.is_integer_dtype(data_nonan) and (data_nonan >= 0).all():
        app.logger.info("Detected non-negative integer data. Selecting Poisson family.")
        return sm.families.Poisson(), "Poisson"
        
    # Check for Positive Continuous Data (heuristic for Gamma)
    if pd.api.types.is_numeric_dtype(data_nonan) and (data_nonan > 0).all():
         app.logger.info("Detected positive continuous data. Selecting Gamma family.")
         return sm.families.Gamma(), "Gamma"
         
    # Default to Gaussian
    app.logger.info("Data type doesn't strongly match Binomial, Poisson, or Gamma. Defaulting to Gaussian.")
    return sm.families.Gaussian(), "Gaussian (Default)"

# --- Helper function for Backward Selection based on AIC (Updated for GEE) ---
def select_model_aic(df, dependent_var, candidate_predictors, model_type, **kwargs):
    """Performs backward selection using AIC (or QIC for GEE, approximated by AIC here)."""
    app.logger.info(f"Starting AIC selection for {model_type.upper()} with predictors: {candidate_predictors}")
    app.logger.info(f"Model kwargs received: {kwargs}")

    # Ensure grouping variable is present in kwargs for LMM/GEE if needed by the model func
    groups_kwarg = kwargs.get('groups', None)
    family_kwarg = kwargs.get('family', None)
    cov_struct_kwarg = kwargs.get('cov_struct', None) # For GEE
    re_formula_kwarg = kwargs.get('re_formula', None) # For LMM

    # Map model type to the correct statsmodels formula function
    model_func_map = {
        'lm': smf.ols,
        'glm': smf.glm,
        'lmm': smf.mixedlm,
        'glmm': smf.gee # Treat 'glmm' as GEE here
    }
    model_func = model_func_map.get(model_type)
    if not model_func:
        raise ValueError(f"Unsupported model type for AIC selection: {model_type}")

    current_predictors = list(candidate_predictors)
    best_model = None
    best_aic = float('inf')
    best_formula = None
    selection_log = [f"Starting AIC-based backward selection for {model_type.upper()}."]
    selection_log.append(f"Initial predictors: {', '.join(current_predictors)}")

    iteration = 0
    max_iterations = len(current_predictors) + 1 # Safety break

    while len(current_predictors) > 0 and iteration < max_iterations:
        iteration += 1
        selection_log.append(f"\nIteration {iteration} - Current predictors: {', '.join(current_predictors)}")
        models_this_iter = []

        # Fit model with all current predictors
        formula = f"{dependent_var} ~ {' + '.join(current_predictors)}"
        current_kwargs = {}
        if family_kwarg: current_kwargs['family'] = family_kwarg
        if model_type == 'lmm':
            if groups_kwarg is None: raise ValueError("LMM requires 'groups' kwarg for AIC selection.")
            current_kwargs['groups'] = groups_kwarg
            if re_formula_kwarg: current_kwargs['re_formula'] = re_formula_kwarg
        elif model_type == 'glmm': # GEE specific kwargs
            if groups_kwarg is None: raise ValueError("GEE (as GLMM) requires 'groups' kwarg for AIC selection.")
            if cov_struct_kwarg is None: raise ValueError("GEE (as GLMM) requires 'cov_struct' kwarg for AIC selection.")
            current_kwargs['groups'] = groups_kwarg
            current_kwargs['cov_struct'] = cov_struct_kwarg
        
        try:
            app.logger.info(f"Fitting full model ({model_type}) this iteration: {formula} with kwargs: {current_kwargs}")
            full_model = model_func(formula, data=df, **current_kwargs).fit()
            # Use AIC for all models for simplicity, though QIC is preferred for GEE
            # Note: statsmodels GEE results don't directly have .aic, need calculation or alternative
            # Let's try a placeholder/skip AIC check if model is GEE for now, or find QIC later
            if model_type == 'glmm':
                # GEE doesn't have .aic attribute. Using BIC as a proxy or skipping AIC check might be needed.
                # For now, let's just fit the full model and assume it's the best if no predictors are removed.
                # A proper QIC implementation would be better.
                current_aic = 0 # Placeholder - GEE AIC/QIC needs specific handling
                selection_log.append(f"  Full model ({formula}): AIC/QIC not directly available from statsmodels GEE result. Proceeding.")
                # If it's the first iteration and GEE, store this as the potential best model
                if best_model is None:
                     best_model = full_model
                     best_formula = formula
                     # best_aic remains inf, so any predictor removal check won't improve it based on AIC=0
            else:
                 current_aic = full_model.aic
                 selection_log.append(f"  Full model ({formula}): AIC = {current_aic:.4f}")
            
            # Store the full model of this iteration if it's the best seen so far (or first)
            # For GEE, this logic needs refinement based on QIC or another criterion
            if best_model is None or (model_type != 'glmm' and current_aic < best_aic):
                best_aic = current_aic
                best_model = full_model
                best_formula = formula

        except Exception as e:
            app.logger.error(f"Failed to fit full model in AIC iteration {iteration} ({formula}): {e}", exc_info=True)
            selection_log.append(f"  ERROR fitting full model: {e}")
            # If the full model fails, we can't proceed with this set of predictors
            raise ValueError(f"AIC selection failed: Could not fit model with predictors {', '.join(current_predictors)}. Error: {e}")

        # Try removing each predictor one by one
        aic_if_removed = {}
        if len(current_predictors) > 1: # Only try removing if more than one predictor
            for predictor_to_remove in current_predictors:
                temp_predictors = [p for p in current_predictors if p != predictor_to_remove]
                formula_reduced = f"{dependent_var} ~ {' + '.join(temp_predictors)}"
                try:
                    # Use the same kwargs as the full model fit
                    app.logger.info(f"  Trying removal of {predictor_to_remove}: {formula_reduced}")
                    reduced_model = model_func(formula_reduced, data=df, **current_kwargs).fit()
                    if model_type == 'glmm':
                        # Placeholder for GEE QIC/AIC
                        aic_val = 0 # Needs proper QIC calculation
                        selection_log.append(f"    Model without '{predictor_to_remove}': AIC/QIC = N/A (GEE)")
                    else:
                        aic_val = reduced_model.aic
                        selection_log.append(f"    Model without '{predictor_to_remove}': AIC = {aic_val:.4f}")
                    aic_if_removed[predictor_to_remove] = aic_val

                except Exception as e:
                    app.logger.warning(f"Could not fit model after removing {predictor_to_remove}: {e}")
                    selection_log.append(f"    ERROR fitting model without '{predictor_to_remove}': {e}")
                    # Treat failure to fit as a very high AIC to avoid removing this predictor
                    aic_if_removed[predictor_to_remove] = float('inf')
        else:
             selection_log.append("  Only one predictor left, cannot remove further.")


        # Find predictor whose removal results in the lowest AIC (or QIC proxy)
        # For GEE, this comparison is currently broken due to placeholder AIC=0
        # We need QIC or a different strategy for GEE variable selection
        
        # --- TEMPORARY GEE Strategy: Stop selection if GEE, return full model ---
        if model_type == 'glmm':
             selection_log.append("\nAIC/QIC selection for GEE is not fully implemented. Returning the initial model.")
             if best_model is None: # Should have been set in the first 'try' block
                 raise ValueError("AIC selection failed for GEE: Initial model could not be fitted.")
             # Use the formula from the first successfully fitted model
             best_formula = f"{dependent_var} ~ {' + '.join(candidate_predictors)}" 
             selection_log.append(f"Selected model (GEE - Initial Full Model): {best_formula}")
             return best_model, best_formula, "\n".join(selection_log)
        # --- End Temporary GEE Strategy ---


        predictor_to_drop = None
        lowest_aic_on_removal = best_aic # Start with the AIC of the model including all current predictors

        if aic_if_removed: # If we tested removals
             sorted_removals = sorted(aic_if_removed.items(), key=lambda item: item[1])
             # Check if the best AIC after removing a variable is lower than the current best AIC
             if sorted_removals[0][1] < lowest_aic_on_removal:
                 predictor_to_drop = sorted_removals[0][0]
                 lowest_aic_on_removal = sorted_removals[0][1] # Update the best AIC found

        if predictor_to_drop:
            current_predictors.remove(predictor_to_drop)
            selection_log.append(f"  Removed '{predictor_to_drop}' (New best AIC: {lowest_aic_on_removal:.4f})")
            best_aic = lowest_aic_on_removal # Update overall best AIC
            # Refit the best model found after removal to store it
            best_formula = f"{dependent_var} ~ {' + '.join(current_predictors)}"
            try:
                 best_model = model_func(best_formula, data=df, **current_kwargs).fit()
                 selection_log.append(f"  Current best model formula: {best_formula}")
            except Exception as e:
                 app.logger.error(f"Failed to refit best model after removing {predictor_to_drop}: {e}", exc_info=True)
                 selection_log.append(f"  ERROR refitting model after removal: {e}")
                 # This indicates a potential problem, maybe stop selection?
                 raise ValueError(f"AIC selection failed: Could not refit model after removing {predictor_to_drop}. Error: {e}")

        else:
            selection_log.append("  No predictor removal improved AIC. Stopping selection.")
            break # Exit while loop

    if best_model is None:
         # This might happen if the very first model failed, or potentially with GEE logic issues
         raise ValueError(f"AIC selection failed for {model_type.upper()}. The initial full model likely failed to fit/converge or GEE selection logic failed. Check logs.")
    
    selection_log.append(f"\nBackward selection finished.")
    selection_log.append(f"Selected model ({model_type.upper()}): {best_formula} (AIC: {best_aic:.4f})") # AIC might be inaccurate for GEE

    return best_model, best_formula, "\n".join(selection_log)

# --- Helper function to generate effect plots (Updated for GEE) ---
def generate_effect_plots(model, df, dependent_var, predictors, unique_id):
    """Generates enhanced effect plots for each predictor."""
    effect_plots_data = [] # List to store dicts {url: ..., interpretation: ...}
    
    # Font properties - attempt Times New Roman, fallback to serif
    font_family = 'Times New Roman' # Desired font
    # Check if font is available (basic check)
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if font_family not in available_fonts:
         app.logger.warning(f"Font '{font_family}' not found. Using default serif font.")
         font_family = 'serif' # Default fallback
         
    plot_rc_params = {
        'font.family': font_family,
        'axes.labelsize': 14, 
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.titlesize': 16
    }

    numeric_predictors = df[predictors].select_dtypes(include=np.number).columns.tolist()
    categorical_predictors = df[predictors].select_dtypes(exclude=np.number).columns.tolist()

    constant_data = {}
    for p in numeric_predictors:
        constant_data[p] = df[p].mean()
    for p in categorical_predictors:
        try: constant_data[p] = df[p].mode()[0] 
        except IndexError: constant_data[p] = df[p].iloc[0] # Fallback if mode fails

    for predictor in predictors:
        interpretation_text = ""
        plot_generated = False
        try:
            with plt.rc_context(plot_rc_params):
                plt.figure(figsize=(7, 7)) 
                ax = plt.gca()
                n_points = 100
                pred_df = pd.DataFrame([constant_data] * n_points)
                plot_title = f"Effect of {predictor}"
                y_label = f"Predicted {model.model.endog_names}"
                
                # --- Check model type for prediction method (LMM only) ---
                is_lmm = isinstance(model, statsmodels.regression.mixed_linear_model.MixedLMResults)
                is_gee = isinstance(model, statsmodels.genmod.generalized_estimating_equations.GEEResultsWrapper)
                app.logger.info(f"Generating plots. Is LMM? {is_lmm}. Is GEE? {is_gee}")

                if predictor in numeric_predictors:
                    x_values = np.linspace(df[predictor].min(), df[predictor].max(), n_points)
                    pred_df[predictor] = x_values
                    
                    mean_pred = None
                    ci_lower = None
                    ci_upper = None
                    legend_text = ""

                    if is_lmm:
                        app.logger.info("Using LMM predict() method.")
                        try:
                             design_info = model.model.data.design_info
                             exog_pred = patsy.dmatrix(design_info, pred_df, return_type='dataframe')
                             mean_pred = model.predict(exog_pred)
                             legend_text = "\n\n**Legend:** Black dots = Original Data; Black line = Predicted Mean (Population Level - CI not shown)."
                        except Exception as mm_pred_e:
                             app.logger.error(f"Failed to get LMM prediction for {predictor}: {mm_pred_e}", exc_info=True)
                             flash(f"Error getting LMM prediction for {predictor}. Skipping effect plot.")
                             continue
                    elif is_gee:
                        app.logger.info("Using GEE predict() method.")
                        try:
                            # GEE predict gives mean prediction for the provided exog
                            mean_pred = model.predict(pred_df)
                            # GEE results don't easily provide CI for predictions like GLM. 
                            # We'll plot only the mean line.
                            legend_text = "\n\n**Legend:** Black dots = Original Data; Black line = Predicted Mean (Population Level - CI not available)."
                        except Exception as gee_pred_e:
                             app.logger.error(f"Failed to get GEE prediction for {predictor}: {gee_pred_e}", exc_info=True)
                             flash(f"Error getting GEE prediction for {predictor}. Skipping effect plot.")
                             continue
                    else: # LM or GLM
                        app.logger.info("Using get_prediction() method.")
                        try:
                            pred_summary = model.get_prediction(pred_df).summary_frame()
                            mean_pred = pred_summary['mean']
                            ci_lower = pred_summary['mean_ci_lower']
                            ci_upper = pred_summary['mean_ci_upper']
                            legend_text = "\n\n**Legend:** Black dots = Original Data; Black line = Predicted Mean; Grey area = 95% Confidence Interval."
                        except Exception as gp_e:
                            app.logger.error(f"Failed to get OLS/GLM prediction for {predictor}: {gp_e}", exc_info=True)
                            flash(f"Error getting prediction for {predictor}. Skipping effect plot.")
                            continue # Skip to next predictor

                    # Ensure mean_pred was successfully calculated
                    if mean_pred is None:
                         app.logger.warning(f"Mean prediction is None for {predictor}. Skipping plot.")
                         continue
                         
                    # Plot original data points
                    ax.scatter(df[predictor], df[dependent_var], s=25, color='black') 
                    # Plot prediction line
                    ax.plot(x_values, mean_pred, color='black')
                    # Plot CI only if available (not LMM in this implementation)
                    if ci_lower is not None and ci_upper is not None:
                        ax.fill_between(x_values, ci_lower, ci_upper, color='grey', alpha=0.3)
                        
                    ax.set_xlabel(predictor)
                    ax.set_ylabel(y_label)
                    ax.set_title(plot_title)
                    
                    # Generate interpretation (use mean_pred)
                    effect_interpretation = ""
                    # Ensure mean_pred is index-aligned with x_values if it came from LMM predict
                    if isinstance(mean_pred, pd.Series):
                        slope = (mean_pred.iloc[-1] - mean_pred.iloc[0]) / (x_values[-1] - x_values[0]) if len(mean_pred)>1 else 0
                    else: # Assuming numpy array
                        slope = (mean_pred[-1] - mean_pred[0]) / (x_values[-1] - x_values[0]) if len(mean_pred)>1 else 0
                        
                    if abs(slope) > 1e-6:
                         direction = "positive" if slope > 0 else "negative"
                         effect_interpretation = f"The plot suggests a {direction} relationship between `{predictor}` and the predicted `{dependent_var}`. As `{predictor}` increases, the prediction tends to {'increase' if direction == 'positive' else 'decrease'}, holding other variables constant."
                    else:
                         effect_interpretation = f"The plot suggests little to no linear relationship between `{predictor}` and the predicted `{dependent_var}` across its range, holding other variables constant."
                    
                    interpretation_text = effect_interpretation + legend_text
                    plot_generated = True

                elif predictor in categorical_predictors:
                    levels = df[predictor].unique()
                    pred_df_cat = pd.DataFrame([constant_data] * len(levels))
                    pred_df_cat[predictor] = levels
                    
                    mean_pred_cat = None
                    legend_text_cat = ""
                    pred_summary_cat_df = None # For plotting
                    errorbar_arg = None # For pointplot

                    if is_lmm:
                        app.logger.info("Using LMM predict() for categorical.")
                        try:
                            design_info = model.model.data.design_info
                            exog_pred_cat = patsy.dmatrix(design_info, pred_df_cat, return_type='dataframe')
                            mean_pred_cat = model.predict(exog_pred_cat)
                            pred_summary_cat_df = pd.DataFrame({'mean': mean_pred_cat, predictor: levels})
                            legend_text_cat = "\n\n**Legend:** Black dots = Predicted Mean (Population Level - CI not shown)."
                            errorbar_arg = None # No CI for LMM pointplot
                        except Exception as mm_pred_e_cat:
                            app.logger.error(f"Failed to get LMM categorical prediction for {predictor}: {mm_pred_e_cat}", exc_info=True)
                            flash(f"Error getting LMM categorical prediction for {predictor}. Skipping effect plot.")
                            continue
                    elif is_gee:
                        app.logger.info("Using GEE predict() for categorical.")
                        try:
                            # GEE predict gives mean prediction
                            mean_pred_cat = model.predict(pred_df_cat)
                            # Convert to DataFrame for plotting with seaborn
                            plot_data_cat = pd.DataFrame({predictor: levels, 'predicted_mean': mean_pred_cat})
                            errorbar_arg = None # No CI easily available
                            legend_text_cat = "\n\n**Legend:** Black dots = Predicted Mean (Population Level - CI not available)."
                        except Exception as gee_pred_cat_e:
                            app.logger.error(f"Failed to get GEE prediction for categorical {predictor}: {gee_pred_cat_e}", exc_info=True)
                            flash(f"Error getting GEE categorical prediction for {predictor}. Skipping plot.")
                            continue
                    else: # LM or GLM
                        app.logger.info("Using get_prediction() for categorical.")
                        try:
                            pred_summary_cat = model.get_prediction(pred_df_cat).summary_frame()
                            pred_summary_cat_df = pred_summary_cat # Already has mean, CIs etc.
                            pred_summary_cat_df[predictor] = levels 
                            legend_text_cat = "\n\n**Legend:** Black dots = Predicted Mean; Vertical lines = 95% Confidence Interval."
                            errorbar_arg = ('ci', 95) # Use standard CI for pointplot
                        except Exception as gp_e_cat:
                            app.logger.error(f"Failed to get OLS/GLM categorical prediction for {predictor}: {gp_e_cat}", exc_info=True)
                            flash(f"Error getting categorical prediction for {predictor}. Skipping effect plot.")
                            continue

                    if pred_summary_cat_df is None:
                        app.logger.warning(f"Categorical prediction dataframe is None for {predictor}. Skipping plot.")
                        continue
                        
                    sns.pointplot(x=predictor, y='mean', data=pred_summary_cat_df, color='black', errorbar=errorbar_arg, join=False, capsize=.2, ax=ax)
                    ax.set_xlabel(predictor)
                    ax.set_ylabel(y_label)
                    ax.set_title(plot_title)
                    plt.xticks(rotation=45, ha='right')
                    
                    # Interpretation (remains the same logic, uses pred_summary_cat_df)
                    min_level = pred_summary_cat_df.loc[pred_summary_cat_df['mean'].idxmin()][predictor]
                    max_level = pred_summary_cat_df.loc[pred_summary_cat_df['mean'].idxmax()][predictor]
                    effect_interpretation = f"The plot shows the predicted mean of `{dependent_var}` for each level of `{predictor}`. The highest prediction is for level `{max_level}` and the lowest for level `{min_level}`, holding other variables constant. Check confidence intervals for overlap to assess significance of differences."
                    interpretation_text = effect_interpretation + legend_text_cat
                    plot_generated = True

            if plot_generated:
                plot_base_name = f"effect_{predictor}_{unique_id}"
                png_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.png")
                svg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.svg")
                jpg_path = os.path.join(PLOT_FOLDER, f"{plot_base_name}.jpg")
                
                plt.tight_layout()
                plt.savefig(png_path, dpi=500, bbox_inches='tight')
                plt.savefig(svg_path, bbox_inches='tight')
                plt.savefig(jpg_path, dpi=500, bbox_inches='tight') # Save JPEG with high resolution
                plt.close()
                
                effect_plots_data.append({
                    'key': f'effect_{predictor}', # Keep a simple key if needed
                    'predictor': predictor,
                    'plot_type_for_download': predictor, # Use predictor name for download route
                    'url': url_for('static', filename=f'plots/{plot_base_name}.png'), # PNG for display
                    'interpretation': interpretation_text
                })
                app.logger.info(f"Saved effect plot for {predictor} (png, svg, jpg) for {unique_id}")

        except Exception as e:
            flash(f"Warning: Could not generate enhanced effect plot for '{predictor}'. Error: {e}")
            app.logger.error(f"Error generating effect plot for {predictor}: {e}", exc_info=True)
            os.makedirs(PLOT_FOLDER, exist_ok=True)

    return effect_plots_data

# --- Helper: Generate Basic Interpretation (with enhanced logging) ---
def interpret_results(model_results, model_description, final_predictors):
    """Generates a basic textual interpretation of model results."""
    app.logger.info("Starting results interpretation...")
    interpretation = [f"**Model Interpretation:**\n", f"- **Model Type:** {model_description}\n"]
    
    try:
        # Get p-values and coefficients
        app.logger.info("Accessing model p-values and parameters...")
        p_values = model_results.pvalues
        coeffs = model_results.params
        app.logger.info(f"P-values index: {p_values.index}")
        app.logger.info(f"Coeffs index: {coeffs.index}")
        app.logger.info(f"Final predictors list for interpretation: {final_predictors}")
        
        significant_predictors = []
        # Check significance (excluding intercept if present)
        for predictor in final_predictors:
             app.logger.debug(f"Interpreting predictor: {predictor}")
             # Handle potential issues if predictor not in results (e.g., after selection)
             if predictor in p_values.index and predictor in coeffs.index:
                 p_val = p_values[predictor]
                 coeff_val = coeffs[predictor]
                 app.logger.debug(f"  Found {predictor}: p={p_val}, coeff={coeff_val}")
                 if p_val < 0.05:
                     # Determine direction based on coefficient sign
                     coeff_direction = "positive" if coeff_val > 0 else "negative"
                     
                     # --- Determine direction on original scale based on link function --- 
                     link_function_name = "identity" # Default assumption (e.g., for LM)
                     try:
                         # Access link function info (may vary slightly between model types)
                         if hasattr(model_results, 'model') and hasattr(model_results.model, 'family') and hasattr(model_results.model.family, 'link'):
                            link_obj = model_results.model.family.link
                            # Get the class name of the link function
                            link_function_name = link_obj.__class__.__name__.lower()
                            app.logger.info(f"[Formal Results] Predictor: {predictor}, Link function detected: {link_function_name}")
                         else:
                             app.logger.info(f"[Formal Results] Predictor: {predictor}, Could not detect link function, assuming identity.")
                     except Exception as link_err:
                          app.logger.warning(f"[Formal Results] Error detecting link function for {predictor}: {link_err}")
                     
                     # Translate direction to original scale
                     original_scale_direction = coeff_direction
                     if link_function_name in ['inversepower', 'inverse_squared']: # Add other inverse links if needed
                         original_scale_direction = "positive" if coeff_direction == "negative" else "negative"
                         app.logger.info(f"[Formal Results] Link ({link_function_name}) reverses effect direction for {predictor}.")
                     # For log, logit, identity etc., the direction is the same
                     
                     # --- DEBUG LOG --- 
                     app.logger.info(f"[Formal Results] Predictor: {predictor}, Coeff: {coeff_val:.4f}, Coeff_Direction: {coeff_direction}, Original_Scale_Direction: {original_scale_direction}")
                     # --- END DEBUG LOG --- 
                     
                     plot_ref = f"(see effect plot for {predictor})"
                     
                     # Use original_scale_direction in the output text
                     text.append(f"- **{predictor}** had a significant **{original_scale_direction}** effect on `{dependent_var}` (p = {p_val:.3f}) {plot_ref}. ")
             else:
                 app.logger.warning(f"Predictor '{predictor}' not found in final model results index (p-values or coeffs) for interpretation.")

        if significant_predictors:
            # The surrounding text already uses Markdown bold
            interpretation.append("- **Significant Predictors (p < 0.05):** " + ", ".join(significant_predictors) + ".\n")
            
        # Add R-squared / Pseudo R-squared (with extra error handling)
        app.logger.info("Checking for R-squared / Pseudo R-squared...")
        try:
            if isinstance(model_results, sm.regression.linear_model.RegressionResultsWrapper):
                 app.logger.info("Model is LM type. Accessing rsquared...")
                 rsq = model_results.rsquared
                 interpretation.append(f"- **Model Fit (R-squared):** {rsq:.3f}\n")
                 app.logger.info(f"Added R-squared: {rsq:.3f}")
            elif hasattr(model_results, 'pseudo_rsquared'):
                 app.logger.info("Model has pseudo_rsquared attribute. Attempting calculation...")
                 try:
                     pseudo_r2 = model_results.pseudo_rsquared()
                     if pseudo_r2 is not None:
                         interpretation.append(f"- **Model Fit (Pseudo R-squared):** {pseudo_r2:.3f}\n")
                         app.logger.info(f"Added Pseudo R-squared: {pseudo_r2:.3f}")
                     else:
                         app.logger.info("pseudo_rsquared() returned None.")
                 except Exception as prs_e:
                     app.logger.warning(f"Could not calculate pseudo R-squared: {prs_e}")
                     pass # Ignore if pseudo r-squared calc fails
            else:
                 app.logger.info("Model type does not support R-squared or pseudo R-squared in this check.")
        except AttributeError as ae:
             app.logger.warning(f"AttributeError accessing R-squared/pseudo R-squared: {ae}")
             # Optionally add a note about missing fit metric?
             pass # Continue without the fit metric if specific access fails
        except Exception as fit_metric_e:
             app.logger.error(f"Unexpected error accessing fit metric: {fit_metric_e}", exc_info=True)
             # Optionally add a note about missing fit metric?
             pass # Continue without the fit metric on unexpected error
        
        interpretation.append("\n*Note: This is a basic automated interpretation. Always consider effect sizes, model assumptions, diagnostic plots, and the specific context of your research.*")
        app.logger.info("Interpretation generation successful.")
        
    except Exception as e:
        app.logger.error(f"Error during results interpretation: {e}", exc_info=True) # Log traceback
        interpretation.append("\n*Error occurred during automated interpretation generation.*")

    return "".join(interpretation)

# --- NEW: Generate Formal Results Text --- 
def generate_formal_results_text(df, model_results, model_description, dependent_var, final_predictors, effect_plots_data):
    """Generates a more formal, narrative summary of the results."""
    app.logger.info("Starting formal results text generation...")
    text = []

    try:
        n_obs = len(df)
        text.append(f"### Data Overview\n\nA total of {n_obs} data points were analyzed, examining the relationship between `{dependent_var}` and potential predictors.\n")

        text.append(f"### Statistical Analysis\n\nThe analysis employed a {model_description}. ")
        # Add selection method info if applicable (needs info passed)
        # if aic_selected: text.append("Model selection using AIC/QIC backward elimination identified the final set of predictors. ") 
        # else: text.append("The full model including all specified predictors was fitted. ")
        text.append("Significance was assessed at p < 0.05.\n")

        text.append("### Key Findings\n")
        
        # Get p-values and coefficients
        p_values = model_results.pvalues
        coeffs = model_results.params
        significant_found = False
        
        for predictor in final_predictors:
            if predictor in p_values.index and predictor in coeffs.index:
                p_val = p_values[predictor]
                coeff_val = coeffs[predictor]
                if p_val < 0.05:
                    significant_found = True
                    # Determine direction based on coefficient sign
                    coeff_direction = "positive" if coeff_val > 0 else "negative"
                    
                    # --- Determine direction on original scale based on link function --- 
                    link_function_name = "identity" # Default assumption (e.g., for LM)
                    try:
                        # Access link function info (may vary slightly between model types)
                        if hasattr(model_results, 'model') and hasattr(model_results.model, 'family') and hasattr(model_results.model.family, 'link'):
                           link_obj = model_results.model.family.link
                           # Get the class name of the link function
                           link_function_name = link_obj.__class__.__name__.lower()
                           app.logger.info(f"[Formal Results] Predictor: {predictor}, Link function detected: {link_function_name}")
                        else:
                            app.logger.info(f"[Formal Results] Predictor: {predictor}, Could not detect link function, assuming identity.")
                    except Exception as link_err:
                         app.logger.warning(f"[Formal Results] Error detecting link function for {predictor}: {link_err}")
                     
                    # Translate direction to original scale
                    original_scale_direction = coeff_direction
                    if link_function_name in ['inversepower', 'inverse_squared']: # Add other inverse links if needed
                        original_scale_direction = "positive" if coeff_direction == "negative" else "negative"
                        app.logger.info(f"[Formal Results] Link ({link_function_name}) reverses effect direction for {predictor}.")
                    # For log, logit, identity etc., the direction is the same
                    
                    # --- DEBUG LOG --- 
                    app.logger.info(f"[Formal Results] Predictor: {predictor}, Coeff: {coeff_val:.4f}, Coeff_Direction: {coeff_direction}, Original_Scale_Direction: {original_scale_direction}")
                    # --- END DEBUG LOG --- 
                    
                    plot_ref = f"(see effect plot for {predictor})"
                    
                    # Use original_scale_direction in the output text
                    text.append(f"- **{predictor}** had a significant **{original_scale_direction}** effect on `{dependent_var}` (p = {p_val:.3f}) {plot_ref}. ")
        
        if not significant_found:
             text.append("- No predictors showed a statistically significant association with `{dependent_var}` (at p < 0.05) in the final model.\n")
        else:
            text.append("\n") # Add space after listing predictors
            
        # Add Model Fit (reuse logic from interpret_results if needed)
        try:
            fit_metric_added = False
            if isinstance(model_results, sm.regression.linear_model.RegressionResultsWrapper):
                 rsq = model_results.rsquared
                 text.append(f"The overall model fit, as indicated by R-squared, was {rsq:.3f}.\n")
                 fit_metric_added = True
            elif hasattr(model_results, 'pseudo_rsquared'):
                 pseudo_r2 = model_results.pseudo_rsquared()
                 if pseudo_r2 is not None:
                     text.append(f"The overall model fit, as indicated by Pseudo R-squared, was {pseudo_r2:.3f}.\n")
                     fit_metric_added = True
            # if fit_metric_added: text.append("\n")
        except Exception as fit_e:
             app.logger.warning(f"Could not add fit metric to formal results: {fit_e}")

        text.append("\n*Disclaimer: This is an automated summary based on the statistical model output. It requires careful review, consideration of effect sizes, model assumptions (verified via diagnostic plots), and interpretation within the specific ecological context.*")
        
    except Exception as e:
        app.logger.error(f"Error generating formal results text: {e}", exc_info=True)
        return "*Error occurred during automated formal results generation.*"

    return "\n".join(text)

# --- NEW: Function to call Hugging Face API (Accepts token) ---
def get_ai_interpretation(prompt, api_token):
    if not api_token:
        return "AI interpretation disabled: Hugging Face API token not provided or configured."

    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300, # Limit response length
            "return_full_text": False, # Only get the generated part
            "temperature": 0.7 # Control randomness
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=45) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
            return result[0]['generated_text'].strip()
        else:
            app.logger.error(f"Unexpected response format from Hugging Face API: {result}")
            return "Error: Received unexpected format from AI API."
            
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error calling Hugging Face API: {e}")
        error_message = f"Error connecting to AI API (check token?): {e}"
        if hasattr(e, 'response') and e.response is not None:
             try:
                  error_detail = e.response.json()
                  error_message += f" - {error_detail.get('error', '')}"
             except requests.exceptions.JSONDecodeError:
                  error_message += f" - Status: {e.response.status_code}"
        return error_message
    except Exception as e:
        app.logger.error(f"Generic error during AI interpretation call: {e}", exc_info=True)
        return "An unexpected error occurred during AI interpretation."

@app.route('/')
def index():
    # List files in the upload folder to show user
    try:
        uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
        # Filter out hidden files like .DS_Store
        uploaded_files = [f for f in uploaded_files if not f.startswith('.')]
    except FileNotFoundError:
        uploaded_files = []
        flash("Upload directory not found. It will be created on first upload.")
    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = file.filename  # Use werkzeug's secure_filename in production
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath) # Indent this
            flash(f'File "{filename}" uploaded successfully.') # Indent this
            return redirect(url_for('analyze', filename=filename)) # Indent this
        except Exception as e:
            flash(f'An error occurred while saving the file: {e}') # Indent this
            return redirect(url_for('index')) # Indent this
    # Fallback if file is not valid
    return redirect(url_for('index'))

@app.route('/analyze/<filename>', methods=['GET', 'POST'])
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash(f'Error: File "{filename}" not found.')
        return redirect(url_for('index'))

    try:
        # Try reading the file
        try:
            # Try CSV with auto-detection first
            df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='warn')
            # Clean column names for formula usage (replace spaces, special chars)
            df.columns = df.columns.str.replace(r'[\s\.\-\(\)]', '_', regex=True)
            # Handle numeric column names by prefixing (Patsy requirement)
            df.columns = [f'col_{c}' if str(c).isdigit() or not str(c)[0].isalpha() and str(c)[0] != '_' else c for c in df.columns]
            # Ensure uniqueness after cleaning
            df.columns = pd.unique(df.columns)
        except Exception as e_csv:
            try:
                 df = pd.read_excel(filepath)
                 # Clean column names
                 df.columns = df.columns.str.replace(r'[\s\.\-\(\)]', '_', regex=True)
                 # Handle numeric/invalid starting char column names
                 df.columns = [f'col_{c}' if str(c).isdigit() or not str(c)[0].isalpha() and str(c)[0] != '_' else c for c in df.columns]
                 # Ensure uniqueness after cleaning
                 df.columns = pd.unique(df.columns)
            except Exception as e_excel:
                 flash(f"Error reading file '{filename}'. Could not parse as CSV or Excel: CSV Error: {e_csv}, Excel Error: {e_excel}")
                 return redirect(url_for('index'))

        columns = df.columns.tolist()

        if request.method == 'GET':
            # --- Prepare data preview ---
            preview_html = None
            try:
                # Get head as HTML, add Bootstrap table classes
                preview_html = df.head().to_html(classes=['table', 'table-sm', 'table-bordered', 'table-striped'], index=False)
            except Exception as e:
                app.logger.error(f"Error generating data preview for {filename}: {e}")
                flash("Could not generate data preview.", "warning")
                
            # Render the analysis configuration form
            return render_template('analyze.html', 
                                 filename=filename, 
                                 columns=columns,
                                 preview_html=preview_html) # Pass preview HTML
        
        elif request.method == 'POST':
            # --- Process form data --- 
            model_type = request.form.get('model_type')
            dependent_var = request.form.get('dependent_var')
            original_independent_vars = request.form.getlist('independent_vars') # Keep original list for context
            independent_vars = list(original_independent_vars) # Copy to modify
            perform_aic_selection = request.form.get('perform_aic') == 'on'

            if not dependent_var or dependent_var not in df.columns:
                 flash(f"Error: Invalid or missing dependent variable '{dependent_var}'.")
                 return render_template('analyze.html', filename=filename, columns=columns)
            
            if not independent_vars or not all(v in df.columns for v in independent_vars):
                 flash("Error: Invalid or missing independent variable(s).")
                 return render_template('analyze.html', filename=filename, columns=columns)

            # --- Ensure session_id for unique plot filenames ---
            session_id = session.get('_id', str(uuid.uuid4())) # Use Flask session ID or generate UUID
            if '_id' not in session: session['_id'] = session_id # Store if newly generated
            app.logger.info(f"Using session_id: {session_id} for this request.")

            # --- Get Variable Descriptions --- 
            dep_var_desc = request.form.get('dep_var_description', '')
            indep_var_descriptions = {}
            for var in independent_vars: # Use the list of vars selected in the form
                desc = request.form.get(f'desc_{var}', '')
                if desc:
                     indep_var_descriptions[var] = desc
            
            # --- Get API Token from Form (fallback to environment) --- 
            user_api_token = request.form.get('hf_api_token')
            if not user_api_token: # If user left it blank, check environment
                user_api_token = HF_API_TOKEN # Use pre-loaded environment token if available
            
            # --- Prepare model arguments (common and specific) ---
            model_kwargs = {}
            model_description_base = ""
            results_summary = None
            model = None
            plot_urls = {} # Combined plots dict
            selection_log = None
            final_predictors = [] # Track predictors in the final model

            try:
                # --- Get Family Selection ---
                selected_family_name = "Gaussian"
                selected_family_obj = sm.families.Gaussian()
                family_source = ""
                
                if model_type == 'glm':
                    family_choice = request.form.get('glm_family', 'auto')
                    if family_choice == 'auto':
                        dep_var_data = df[dependent_var]
                        selected_family_obj, selected_family_name = auto_select_glm_family(dep_var_data)
                        family_source = "(Auto-Selected)"
                        flash(f"Automatically selected family for GLM: {selected_family_name}")
                    else:
                        selected_family_obj = get_family_object(family_choice)
                        selected_family_name = family_choice.capitalize()
                        family_source = "(Manual)"
                        flash(f"Using manually selected family for GLM: {selected_family_name}")
                    model_kwargs['family'] = selected_family_obj
                    model_description_base = f"Generalized Linear Model (GLM) - Family: {selected_family_name} {family_source}"
                    final_predictors = list(independent_vars)
                
                elif model_type == 'lmm':
                    # LMM doesn't use family in statsmodels `mixedlm`
                    grouping_vars = request.form.getlist('grouping_vars')
                    random_formula_part = request.form.get('random_effects_formula')
                    if not grouping_vars: raise ValueError("Grouping variable(s) required for LMM.")
                    cleaned_independent_vars = [v for v in independent_vars if v not in grouping_vars]
                    if len(cleaned_independent_vars) != len(independent_vars): flash("Warning: Grouping variable(s) removed from fixed effects.")
                    if not cleaned_independent_vars: raise ValueError("No fixed effects left after removing grouping variable(s).")
                    independent_vars = cleaned_independent_vars
                    final_predictors = list(independent_vars)
                    model_kwargs['groups'] = df[grouping_vars[0]]
                    if len(grouping_vars) > 1: flash("Warning: Using only the first grouping variable.")
                    if random_formula_part: model_kwargs['re_formula'] = f"~{random_formula_part}"
                    model_description_base = f"Linear Mixed Model (LMM) - Group(s): {', '.join(grouping_vars)}"
                    if random_formula_part: model_description_base += f", Random Effects: ~{random_formula_part}"
                    final_predictors = list(independent_vars) # Make sure final_predictors is set
                
                elif model_type == 'glmm': # Interpret as GEE
                    # GEE uses family
                    family_choice = request.form.get('gee_family', 'auto')
                    if family_choice == 'auto':
                        dep_var_data = df[dependent_var]
                        selected_family_obj, selected_family_name = auto_select_glm_family(dep_var_data)
                        family_source = "(Auto-Selected)"
                        flash(f"Automatically selected family for GEE: {selected_family_name}")
                    else:
                        selected_family_obj = get_family_object(family_choice)
                        selected_family_name = family_choice.capitalize()
                        family_source = "(Manual)"
                        flash(f"Using manually selected family for GEE: {selected_family_name}")
                    model_kwargs['family'] = selected_family_obj
                    
                    # ... (Rest of GEE setup: groups, cov_struct, predictors) ...
                    grouping_vars = request.form.getlist('grouping_vars')
                    if not grouping_vars: raise ValueError("Grouping variable(s) required for GEE (selected as GLMM).")
                    # GEE 'groups' argument can be the column data directly
                    model_kwargs['groups'] = df[grouping_vars[0]] 
                    
                    # Get Covariance Structure from form
                    cov_struct_name = request.form.get('gee_cov_struct', 'exchangeable') # Default to exchangeable
                    selected_cov_struct_obj = get_cov_struct_object(cov_struct_name)
                    model_kwargs['cov_struct'] = selected_cov_struct_obj
                    flash(f"Using covariance structure for GEE: {cov_struct_name.capitalize()}")
                    
                    # ... (clean predictors as before) ...
                    final_predictors = list(independent_vars)
                    
                    model_description_base = f"Generalized Estimating Equations (GEE) - Group(s): {', '.join(grouping_vars)}, Family: {selected_family_name} {family_source}, Covariance: {cov_struct_name.capitalize()}"

                else: # LM or unknown
                    if model_type == 'lm':
                         model_description_base = "Linear Model (OLS)"
                         final_predictors = list(independent_vars) # Make sure final_predictors is set
                    else:
                         raise ValueError(f'Unknown model type selected: {model_type}')
                
                # --- Model Fitting / Selection --- 
                base_formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
                model_description = ""; results_summary = None; model = None; plot_urls = {}; effect_plots = []; selection_log = None
                model_func = None # Initialize

                # Determine model function based on type
                if model_type == 'lm': model_func = smf.ols
                elif model_type == 'glm': model_func = smf.glm
                elif model_type == 'lmm': model_func = smf.mixedlm
                elif model_type == 'glmm': model_func = smf.gee # Map to GEE
                else: raise ValueError("Cannot determine model function.")

                if perform_aic_selection and len(independent_vars) > 1:
                     app.logger.info(f"Performing AIC selection for {model_type}...")
                     # Pass all necessary kwargs for the specific model type
                     model, final_formula, selection_log_str = select_model_aic(df, dependent_var, independent_vars, model_type, **model_kwargs)
                     selection_log = selection_log_str # Get the log string
                     # Extract final predictors from the formula selected by AIC
                     try:
                          final_predictors = list(patsy.dmatrix(final_formula.split('~')[1], df, return_type='dataframe').columns)
                          if 'Intercept' in final_predictors: final_predictors.remove('Intercept')
                          # Clean up patsy-generated names if necessary (e.g., C(var)[T.level])
                          # This part might need refinement based on how patsy names things
                          final_predictors = [p.split('[')[0] if '[' in p else p for p in final_predictors] # Basic cleaning
                          final_predictors = list(dict.fromkeys(final_predictors)) # Remove duplicates
                          app.logger.info(f"Predictors selected by AIC: {final_predictors}")
                     except Exception as e:
                          app.logger.error(f"Error extracting predictors from AIC formula '{final_formula}': {e}")
                          flash("Warning: Could not accurately determine final predictors from AIC selection formula. Using initial list.")
                          final_predictors = list(independent_vars) # Fallback
                     
                     model_description = model_description_base + f" (AIC/QIC Selected - Formula: {final_formula})"
                     # Note: AIC value reported by select_model_aic might be 0 or inaccurate for GEE
                     flash("AIC/QIC selection performed (Note: GEE uses QIC, AIC shown for consistency but may not be optimal criterion).", "info")
                else:
                     app.logger.info(f"Fitting full model ({model_type}): {base_formula}")
                     fit_args = {'formula': base_formula, 'data': df}
                     fit_args.update(model_kwargs) # Add family, groups, cov_struct etc.
                     
                     model = model_func(**fit_args).fit()
                     final_predictors = list(independent_vars) # Use all specified predictors
                     model_description = model_description_base + " (Full Model)"
                     flash("Full model fitted (AIC selection not performed).", "info")

                # --- Get Summary, Generate Plots, and Interpret --- 
                app.logger.info(f"Model fitting complete for {model_type}. Generating results...")
                results_summary = model.summary().as_html() if hasattr(model, 'summary') else str(model) # GEE summary might differ
                
                # --- Generate Diagnostic Plots (Restored Call) ---
                # Note: Compatibility with GEE results might vary.
                diagnostic_plot_urls = {}
                try:
                    app.logger.info(f"Generating diagnostic plots for {model_type}...")
                    # Pass session_id for unique filenames
                    diagnostic_plot_urls = generate_diagnostic_plots(model, session_id) 
                except Exception as diag_err:
                    app.logger.error(f"Could not generate diagnostic plots: {diag_err}", exc_info=True)
                    flash("Warning: Could not generate standard diagnostic plots for this model type or due to an error.", 'warning')

                # --- Generate Effect Plots ---
                app.logger.info(f"Final predictors for plotting: {final_predictors}")
                effect_plots_data = [] # Initialize as list
                if final_predictors:
                    try:
                         # Pass session_id for unique filenames
                         effect_plots_data = generate_effect_plots(model, df, dependent_var, final_predictors, session_id)
                         app.logger.info(f"Generated {len(effect_plots_data)} effect plots.")
                    except Exception as effect_err:
                         app.logger.error(f"Could not generate effect plots: {effect_err}", exc_info=True)
                         flash("Warning: Could not generate predictor effect plots due to an error.", 'warning')
                else:
                    app.logger.info("No final predictors to generate effect plots for.")

                # --- Generate Interpretations --- 
                interpretation_text = interpret_results(model, model_description, final_predictors)
                formal_results_text = generate_formal_results_text(df, model, model_description, dependent_var, final_predictors, effect_plots_data)
                
                # --- Prepare Prompt and Call AI API (Pass token) --- 
                ai_interpretation = "AI interpretation was not run or failed."
                if user_api_token: 
                    app.logger.info("Preparing prompt for AI interpretation...")
                    variable_definitions = f"Dependent Variable: `{dependent_var}` means {dep_var_desc or '(Not described)'}.\nIndependent Variables:\n"
                    for var, desc in indep_var_descriptions.items():
                        variable_definitions += f"- `{var}` means {desc}\n"
                    
                    # Extract key results (e.g., significant findings from formal_results_text)
                    significant_findings_text = ""
                    # Basic extraction - could be improved
                    try:
                        findings_section = formal_results_text.split("### Key Findings\n")[1].split("\n*Disclaimer:")[0]
                        significant_findings_text = findings_section.strip()
                    except IndexError:
                        significant_findings_text = "(Could not extract findings automatically)"

                    # Corrected prompt for initial AI interpretation in analyze route
                    # Removed user_prompt and model_summary_text (use results_summary for context if needed, but formal_results_text is better)
                    prompt = f"""You are an ecologist interpreting statistical results.
Context:
- Dependent Variable: `{dependent_var}` ({dep_var_desc or 'No description'})
- Final Predictors Included: {', '.join(final_predictors)}
- Independent Variable Descriptions: {indep_var_descriptions}
- Model Used: {model_description}
- Key Findings (Formal Text Summary):
{formal_results_text}

Task: Provide a brief ecological interpretation of these key findings, considering the variable definitions and the formal summary. Focus on the significant results mentioned. Be concise (1-3 sentences per main finding). State potential ecological reasons or implications if appropriate, but avoid over-interpretation.
Interpretation:"""
                    
                    app.logger.info("Calling AI API for interpretation...")
                    
                    # Pass the user_api_token to the function
                    ai_interpretation = get_ai_interpretation(prompt, user_api_token) 
                    app.logger.info("AI interpretation received.")
                else:
                     ai_interpretation = "AI interpretation disabled: API Key not provided or configured."

                # --- Render results --- 
                return render_template('results.html', 
                                        filename=filename, 
                                        model_description=model_description, 
                                        results_summary=results_summary,
                                        plot_urls=diagnostic_plot_urls, # Pass diagnostic plot dict as plot_urls
                                        effect_plots=effect_plots_data, # Pass effect plot list as effect_plots
                                        interpretation=interpretation_text,
                                        formal_results_text=formal_results_text,
                                        ai_interpretation=ai_interpretation, # Add AI text
                                        selection_log=selection_log, 
                                        session_id=session_id,
                                        # --- Pass additional context for interactive AI --- 
                                        dependent_var=dependent_var,
                                        final_predictors=final_predictors, # List of predictor names
                                        dep_var_desc=dep_var_desc, # String description
                                        indep_var_descriptions=indep_var_descriptions, # Dict {var: desc}
                                        user_api_token=user_api_token # The token used (from form or env)
                                        )

            except (patsy.PatsyError, ValueError, pd.errors.ParserError, KeyError, AttributeError, Exception) as e:
                # Corrected indentation for error handling
                flash(f"Error during analysis: {e}", 'danger')
                if isinstance(e, patsy.PatsyError):
                    flash("PatsyError: Check formula syntax and variable names/types.", 'warning')
                elif isinstance(e, ValueError):
                    flash(f"ValueError: {e}", 'warning')
                
                # Log the full traceback
                import traceback
                app.logger.error(f"Analysis Error for {filename}:\n{traceback.format_exc()}")
                
                # Try to return to analyze page with context if possible
                columns_err = [] # Initialize
                try:
                    filepath_err = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    # Try reading just the header row to get columns
                    df_err = pd.read_csv(filepath_err, sep=None, engine='python', on_bad_lines='skip', nrows=0) 
                    columns_err = df_err.columns.tolist()
                except Exception as read_err:
                     app.logger.error(f"Could not re-read headers on error: {read_err}")
                 
                return render_template('analyze.html', filename=filename, columns=columns_err, error=str(e))
                 
    except Exception as e:
        # Outer exception handler
        flash(f"An critical error occurred processing the request for '{filename}': {e}", 'danger')
        import traceback
        app.logger.error(f"Outer Error for {filename}:\n{traceback.format_exc()}")
        return redirect(url_for('index'))

    # Fallback redirect
    return redirect(url_for('index'))

@app.route('/download_plot/<unique_id>/<plot_type>/<format>')
@login_required # Protect download
def download_plot(unique_id, plot_type, format):
    # TODO: Add check if the user is allowed to download this plot (based on original file owner?)
    
    # !!! ORIGINAL CODE FOR FINDING/SENDING THE PLOT NEEDS TO BE HERE !!!
    # Example placeholder (ensure this block is present and indented):
    plot_filename = f"{plot_type}_{unique_id}.{format}"
    plot_path = os.path.join(app.config['PLOT_FOLDER'], plot_filename)
    app.logger.info(f"Attempting to send plot: {plot_path}")
    
    if not os.path.exists(plot_path):
        flash(f"Plot file {plot_filename} not found.", "danger")
        return redirect(url_for('index')) # Placeholder redirect
    
    try:
        return send_from_directory(app.config['PLOT_FOLDER'], plot_filename, as_attachment=True)
    except Exception as e:
        app.logger.error(f"Error sending plot {plot_filename}: {e}")
        flash("Error occurred while trying to download the plot.", "danger")
        return redirect(url_for('index')) # Placeholder redirect
    # pass # Make sure at least this pass is here if the block above is missing

@app.route('/ask_ai', methods=['POST'])
@login_required # Protect AI route
def ask_ai():
    # TODO: Ensure context passed belongs to the current user if necessary
    # ... (rest of ask_ai logic) ...
    pass # Add pass to ensure an indented block

# === Database Initialization Command ===
@app.cli.command('init-db')
def init_db_command():
    """Clear existing data and create new tables."""
    with app.app_context(): # Ensure we are in app context
        db.drop_all() # Drop all tables (Use with caution!)
        db.create_all() # Create tables based on models
    print('Initialized the database.')
    # pass # Added temporarily if the block above was commented/missing
# ======================================

# --- Remove app.run block --- 
# if __name__ == '__main__':
#     import logging
#     logging.basicConfig(level=logging.DEBUG) 
#     # Ensure folders exist
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     os.makedirs(app.config['PLOT_FOLDER'], exist_ok=True)
#     app.run(debug=True, port=5001) # Use a different port if needed 

# === Forms ===
class RegistrationForm(FlaskForm):
    username = StringField('Username', 
                           validators=[DataRequired(), Length(min=2, max=20)])
    email = StringField('Email', 
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', 
                                     validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

    # Custom validators to check if username/email already exists
    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('That username is taken. Please choose a different one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('That email is already registered. Please choose a different one or login.')

class LoginForm(FlaskForm):
    email = StringField('Email', 
                        validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')
# ===========

# --- Folder Configuration --- 
# ... (rest of the app setup) ...

# === Authentication Routes ===
@app.route("/register", methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        with app.app_context(): # Ensure context for db operations
            db.session.add(user)
            db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route("/login", methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get('next') # Redirect back after login if needed
            flash('Login Successful!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route("/logout")
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))
# =========================

# --- Application Routes (Modify these) ---
@app.route('/') # Modify index route
@login_required # Protect index route
def index():
    # TODO: Modify to show files ONLY for the current_user
    # List files in the upload folder (TEMPORARY - Needs user association)
    try:
        # This logic needs complete replacement with DB queries based on current_user.id
        # For now, it will still show all files
        uploaded_files = os.listdir(app.config['UPLOAD_FOLDER'])
        uploaded_files = [f for f in uploaded_files if not f.startswith('.')] 
    except FileNotFoundError:
        uploaded_files = []
        # flash("Upload directory not found...") # Less relevant now
    return render_template('index.html', uploaded_files=uploaded_files)

@app.route('/upload', methods=['POST'])
@login_required # Protect upload route
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url) # Redirect likely back to index
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        # TODO: Save file securely associated with current_user.id
        # Option 1 (Cloud Storage - Recommended): Upload to S3/GCS, store ref in DB
        # Option 2 (Local - Ephemeral): Save to user-specific subfolder or unique name
        # For now, keeps saving globally (PROBLEM)
        filename = file.filename # Needs secure filename and user association
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            # TODO: Add record to DB associating filename/path with current_user.id
            flash(f'File "{filename}" uploaded successfully.')
            # Redirect to analyze, passing user context implicitly via login
            return redirect(url_for('analyze', filename=filename))
        except Exception as e:
            flash(f'An error occurred while saving the file: {e}')
            return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/analyze/<filename>', methods=['GET', 'POST'])
@login_required # Protect analyze route
def analyze(filename):
    # TODO: Check if current_user is allowed to access this filename (query DB)
    # This check is missing, allowing any logged-in user to analyze any file if they know the name
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash(f'Error: File "{filename}" not found or you do not have access.')
        return redirect(url_for('index'))
    
    # ... (rest of the analyze logic remains largely the same for now) ...
    # ... (but plotting functions will need adjustment if paths change) ...
    # Ensure session_id usage is still appropriate or replace with user-based ID for uniqueness
    # session_id = str(current_user.id) # Example: Use user ID for plot naming?
    session_id = session.get('_id', str(uuid.uuid4())) # Keep session ID for now
    if '_id' not in session: session['_id'] = session_id
    # ... (rest of analyze route) ...

# ... (Other routes like download_plot, ask_ai also need @login_required and access checks) ...

@app.route('/download_plot/<unique_id>/<plot_type>/<format>')
@login_required # Protect download
def download_plot(unique_id, plot_type, format):
    # TODO: Add check if the user is allowed to download this plot (based on original file owner?)
    # ... (rest of download logic) ...

@app.route('/ask_ai', methods=['POST'])
@login_required # Protect AI route
def ask_ai():
    # TODO: Ensure context passed belongs to the current user if necessary
    # ... (rest of ask_ai logic) ...
    pass # Add pass to ensure an indented block