"""
BLENDER vs MITSUBA MULTI-IMAGE COMPARISON
==========================================
Comprehensive analysis of rendering differences across multiple test cases

SETUP: Edit the FILE_PATHS dictionary below with your actual file paths
"""

import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict


FILE_PATHS = {
    # Blender images
    'blender': {
        'no_window':   'validation/blender/output/images/blenderimage_0_no_window.tiff',
        'undeformed':  'validation/blender/output/images/blenderimage_0.tiff',
        'deformed':    'validation/blender/output/images/blenderimage_1.tiff',
    },
    # Mitsuba images
    'mitsuba': {
        'no_window':   'validation/mitsuba/output/mitsuba_render_no_window.png',
        'undeformed':  'validation/mitsuba/output/mitsuba_render_0_5MPx.png',
        'deformed':    'validation/mitsuba/output/mitsuba_render_1_5MPx.png',
    }
}

OUTPUT_DIR = 'validation/comparison_analysis'

# For 18x16 squares checkerboard, inner corners = (17, 15)
PATTERN_SIZE = (17, 15)

@dataclass
class ComparisonResult:
    """Store results from a single image comparison"""
    name: str
    corners_a: np.ndarray
    corners_b: np.ndarray
    displacement: np.ndarray
    max_error: float
    mean_error: float
    rms_error: float
    std_error: float
    median_error: float


def preprocess_for_corners(img):
    """Preprocess image to improve corner detection"""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )
    
    return img_blur, img_thresh


def find_corners_robust(img, pattern_size, name="Image", verbose=True):
    """Try multiple methods to find checkerboard corners"""
    if verbose:
        print(f"  Finding corners in {name}...", end=" ")
    
    img_blur, img_thresh = preprocess_for_corners(img)
    
    flag_combinations = [
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        cv2.CALIB_CB_ADAPTIVE_THRESH,
        cv2.CALIB_CB_NORMALIZE_IMAGE,
        None,
    ]
    
    images_to_try = [
        ("original", img),
        ("blurred", img_blur),
        ("thresholded", img_thresh),
    ]
    
    for img_variant_name, img_variant in images_to_try:
        for flags in flag_combinations:
            try:
                if flags is None:
                    ret, corners = cv2.findChessboardCorners(img_variant, pattern_size)
                else:
                    ret, corners = cv2.findChessboardCorners(img_variant, pattern_size, flags)
                
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners_refined = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                    
                    if verbose:
                        print(f"✓ ({img_variant_name})")
                    return ret, corners_refined
            except Exception:
                continue
    
    if verbose:
        print("✗ FAILED")
    return False, None


def compare_two_images(img_a, img_b, name_a, name_b, pattern_size, verbose=True):
    """Compare two images and return displacement statistics"""
    if verbose:
        print(f"\nComparing {name_a} vs {name_b}:")
    
    ret_a, corners_a = find_corners_robust(img_a, pattern_size, name_a, verbose)
    ret_b, corners_b = find_corners_robust(img_b, pattern_size, name_b, verbose)
    
    if not (ret_a and ret_b):
        print(f"  ✗ Failed to find corners in both images")
        return None
    
    displacement = corners_b - corners_a
    displacement_magnitude = np.linalg.norm(displacement, axis=2).flatten()
    
    result = ComparisonResult(
        name=f"{name_a} vs {name_b}",
        corners_a=corners_a,
        corners_b=corners_b,
        displacement=displacement,
        max_error=np.max(displacement_magnitude),
        mean_error=np.mean(displacement_magnitude),
        rms_error=np.sqrt(np.mean(displacement_magnitude**2)),
        std_error=np.std(displacement_magnitude),
        median_error=np.median(displacement_magnitude)
    )
    
    if verbose:
        print(f"  Mean displacement: {result.mean_error:.3f} px")
        print(f"  RMS displacement:  {result.rms_error:.3f} px")
    
    return result


def analyze_all_cases(file_paths: Dict, output_dir: Path, pattern_size=(17, 15)):
    """Analyze all test cases"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(" COMPREHENSIVE BLENDER vs MITSUBA COMPARISON")
    print("="*80)
    
    test_cases = [
        ("no_window", "No Window", "No window in scene"),
        ("undeformed", "Undeformed Window", "Window present, not deformed"),
        ("deformed", "Deformed Window", "Window present and deformed"),
    ]
    
    results = {}
    
    for case_id, case_name, case_desc in test_cases:
        print(f"\n{'='*80}")
        print(f" TEST CASE: {case_name}")
        print(f" {case_desc}")
        print(f"{'='*80}")
        
        blender_path = Path(file_paths['blender'].get(case_id, ''))
        mitsuba_path = Path(file_paths['mitsuba'].get(case_id, ''))
        
        if not blender_path.exists():
            print(f"  Blender image not found: {blender_path}")
            continue
        if not mitsuba_path.exists():
            print(f"  Mitsuba image not found: {mitsuba_path}")
            continue
        
        blender_img = cv2.imread(str(blender_path), cv2.IMREAD_GRAYSCALE)
        mitsuba_img = cv2.imread(str(mitsuba_path), cv2.IMREAD_GRAYSCALE)
        
        if blender_img is None:
            print(f"  Failed to load Blender image: {blender_path}")
            continue
        if mitsuba_img is None:
            print(f"  Failed to load Mitsuba image: {mitsuba_path}")
            continue
        
        print(f"  Loaded: {blender_path.name}")
        print(f"  Loaded: {mitsuba_path.name}")
        
        result = compare_two_images(
            blender_img, mitsuba_img,
            "Blender", "Mitsuba",
            pattern_size,
            verbose=True
        )
        
        if result:
            results[case_id] = result
    
    if not results:
        print("\n No successful comparisons!")
        return None
    
    generate_summary_report(results, output_dir, test_cases)
    generate_comparison_plots(results, output_dir, test_cases)
    generate_displacement_visualizations(results, output_dir, test_cases, pattern_size)
    generate_relative_displacement_analysis(results, output_dir, pattern_size, file_paths)
    return results


def generate_summary_report(results: Dict[str, ComparisonResult], output_dir: Path, test_cases):
    """Generate text summary report with normalized errors"""
    
    report_path = output_dir / 'comparison_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" BLENDER vs MITSUBA COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        baseline = results.get('no_window')
        baseline_rms = baseline.rms_error if baseline else 1.0
        
        f.write("ABSOLUTE ERRORS (pixels):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Test Case':<25} {'Mean':>10} {'RMS':>10} {'Max':>10} {'Median':>10} {'Std':>10}\n")
        f.write("-"*80 + "\n")
        
        for case_id, case_name, _ in test_cases:
            if case_id in results:
                r = results[case_id]
                f.write(f"{case_name:<25} "
                       f"{r.mean_error:>10.3f} "
                       f"{r.rms_error:>10.3f} "
                       f"{r.max_error:>10.3f} "
                       f"{r.median_error:>10.3f} "
                       f"{r.std_error:>10.3f}\n")
        
        f.write("\n\n")
        f.write("NORMALIZED ERRORS (relative to 'No Window' baseline):\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Test Case':<25} {'Mean Ratio':>12} {'RMS Ratio':>12} {'Max Ratio':>12}\n")
        f.write("-"*80 + "\n")
        
        if baseline:
            for case_id, case_name, _ in test_cases:
                if case_id in results:
                    r = results[case_id]
                    mean_ratio = r.mean_error / baseline.mean_error
                    rms_ratio = r.rms_error / baseline.rms_error
                    max_ratio = r.max_error / baseline.max_error
                    
                    f.write(f"{case_name:<25} "
                           f"{mean_ratio:>12.3f} "
                           f"{rms_ratio:>12.3f} "
                           f"{max_ratio:>12.3f}\n")
        else:
            f.write("  (No baseline available for normalization)\n")
        
        f.write("\n\n")
        f.write("INTERPRETATION:\n")
        f.write("-"*80 + "\n")
        f.write("1. BASELINE ('No Window'):\n")
        if baseline:
            f.write(f"   RMS error: {baseline.rms_error:.3f} pixels\n")
            f.write("   This represents the fundamental Blender-Mitsuba rendering difference\n")
            f.write("   due to camera model, FOV, or coordinate system mismatch.\n\n")
        else:
            f.write("   (No baseline available)\n\n")
        
        f.write("2. WINDOW EFFECT:\n")
        if 'no_window' in results and 'undeformed' in results:
            delta = results['undeformed'].rms_error - results['no_window'].rms_error
            f.write(f"   Additional RMS error from window: {delta:.3f} pixels\n")
            if delta > 0:
                f.write(f"   Window increases error by {(delta/baseline.rms_error)*100:.1f}% of baseline\n")
            f.write(f"   This is the optical distortion from the undeformed window.\n\n")
        else:
            f.write("   (Missing data for comparison)\n\n")
        
        f.write("3. DEFORMATION EFFECT:\n")
        if 'undeformed' in results and 'deformed' in results:
            delta = results['deformed'].rms_error - results['undeformed'].rms_error
            f.write(f"   Additional RMS error from deformation: {delta:.3f} pixels\n")
            if baseline and delta > 0:
                f.write(f"   Deformation increases error by {(delta/baseline.rms_error)*100:.1f}% of baseline\n")
            f.write(f"   This is the additional optical distortion from window deformation.\n\n")
        else:
            f.write("   (Missing data for comparison)\n\n")
        
        f.write("4. TOTAL OPTICAL EFFECT:\n")
        if 'no_window' in results and 'deformed' in results:
            total_delta = results['deformed'].rms_error - results['no_window'].rms_error
            f.write(f"   Total additional RMS error: {total_delta:.3f} pixels\n")
            if baseline:
                f.write(f"   Combined optical effects: {(total_delta/baseline.rms_error)*100:.1f}% of baseline\n\n")
        else:
            f.write("   (Missing data for comparison)\n\n")
        
        f.write("\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n")
        if baseline and baseline.rms_error > 5:
            f.write("⚠ BASELINE ERROR IS HIGH (>5 pixels)\n")
            f.write("  → Fix Blender-Mitsuba camera mismatch first before analyzing optical effects\n")
            f.write("  → Check focal length, FOV, sensor size, coordinate system\n")
            f.write("  → Baseline should be <1 pixel for accurate optical validation\n\n")
        elif baseline and baseline.rms_error > 1:
            f.write("⚠ BASELINE ERROR IS MODERATE (1-5 pixels)\n")
            f.write("  → Blender-Mitsuba rendering is close but not perfect\n")
            f.write("  → Optical effects may be measurable but with reduced accuracy\n\n")
        elif baseline:
            f.write("✓ BASELINE ERROR IS LOW (<1 pixel)\n")
            f.write("  → Blender-Mitsuba rendering matches well\n")
            f.write("  → Optical effects can be accurately measured\n\n")
    
    print(f"  ✓ Summary report: {report_path.name}")


def generate_comparison_plots(results: Dict[str, ComparisonResult], output_dir: Path, test_cases):
    """Generate comparison bar charts and line plots"""
    
    # Prepare data
    case_names = []
    mean_errors = []
    rms_errors = []
    max_errors = []
    
    for case_id, case_name, _ in test_cases:
        if case_id in results:
            case_names.append(case_name)
            mean_errors.append(results[case_id].mean_error)
            rms_errors.append(results[case_id].rms_error)
            max_errors.append(results[case_id].max_error)
    
    if not case_names:
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Blender vs Mitsuba: Displacement Error Analysis', fontsize=16, fontweight='bold')
    
    x_pos = np.arange(len(case_names))
    width = 0.35
    
    # 1. Bar chart - Mean vs RMS
    ax = axes[0, 0]
    ax.bar(x_pos - width/2, mean_errors, width, label='Mean', alpha=0.8, color='steelblue')
    ax.bar(x_pos + width/2, rms_errors, width, label='RMS', alpha=0.8, color='coral')
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Displacement Error (pixels)')
    ax.set_title('Mean and RMS Displacement Errors')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(case_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Line plot - Error progression
    ax = axes[0, 1]
    ax.plot(case_names, mean_errors, 'o-', label='Mean', linewidth=2, markersize=8, color='steelblue')
    ax.plot(case_names, rms_errors, 's-', label='RMS', linewidth=2, markersize=8, color='coral')
    ax.plot(case_names, max_errors, '^-', label='Max', linewidth=2, markersize=8, color='forestgreen')
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Displacement Error (pixels)')
    ax.set_title('Error Progression Across Test Cases')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Normalized bar chart (if baseline exists)
    ax = axes[1, 0]
    if 'no_window' in results:
        baseline_rms = results['no_window'].rms_error
        normalized_rms = [rms / baseline_rms for rms in rms_errors]
        
        bars = ax.bar(x_pos, normalized_rms, alpha=0.8, color='mediumpurple')
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline')
        ax.set_xlabel('Test Case')
        ax.set_ylabel('Normalized RMS Error')
        ax.set_title('RMS Error Normalized to "No Window" Baseline')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(case_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, normalized_rms):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}x',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No baseline\navailable', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Normalized Errors (Baseline Missing)')
    
    # 4. Incremental effects
    ax = axes[1, 1]
    if len(case_names) >= 2:
        effects = []
        effect_labels = []
        
        if 'no_window' in results and 'undeformed' in results:
            window_effect = results['undeformed'].rms_error - results['no_window'].rms_error
            effects.append(window_effect)
            effect_labels.append('Window\nEffect')
        
        if 'undeformed' in results and 'deformed' in results:
            deform_effect = results['deformed'].rms_error - results['undeformed'].rms_error
            effects.append(deform_effect)
            effect_labels.append('Deformation\nEffect')
        
        if 'no_window' in results and 'deformed' in results:
            total_effect = results['deformed'].rms_error - results['no_window'].rms_error
            effects.append(total_effect)
            effect_labels.append('Total\nEffect')
        
        if effects:
            colors = ['skyblue', 'lightcoral', 'lightgreen'][:len(effects)]
            bars = ax.bar(range(len(effects)), effects, color=colors, alpha=0.8)
            ax.set_xticks(range(len(effects)))
            ax.set_xticklabels(effect_labels)
            ax.set_ylabel('Additional RMS Error (pixels)')
            ax.set_title('Incremental Optical Effects')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add value labels
            for bar, val in zip(bars, effects):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}px',
                       ha='center', va='bottom' if val > 0 else 'top', 
                       fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor incremental analysis', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Incremental Effects')
    
    plt.tight_layout()
    
    output_path = output_dir / 'comparison_plots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Comparison plots: {output_path.name}")


def generate_displacement_visualizations(results: Dict[str, ComparisonResult], output_dir: Path, test_cases, pattern_size):
    """Generate difference heatmaps between Blender and Mitsuba for each case"""
    
    n_cases = len([c for c in test_cases if c[0] in results])
    if n_cases == 0:
        return
    
    fig, axes = plt.subplots(1, n_cases, figsize=(6*n_cases, 5))
    if n_cases == 1:
        axes = [axes]
    
    col = 0
    for case_id, case_name, _ in test_cases:
        if case_id not in results:
            continue
        
        result = results[case_id]
        displacement_magnitude = np.linalg.norm(result.displacement, axis=2).flatten()
        
        # Reshape into grid
        displacement_grid = displacement_magnitude.reshape(pattern_size[1], pattern_size[0])
        
        ax = axes[col]
        im = ax.imshow(displacement_grid, cmap='hot', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Displacement (px)')
        
        mean_disp = result.mean_error
        max_disp = result.max_error
        rms_disp = result.rms_error
        
        ax.set_title(f'{case_name}\n'
                    f'Mean: {mean_disp:.2f}px | RMS: {rms_disp:.2f}px | Max: {max_disp:.2f}px',
                    fontsize=11)
        ax.set_xlabel('X corner index')
        ax.set_ylabel('Y corner index')
        
        col += 1
    
    plt.tight_layout()
    
    output_path = output_dir / 'blender_mitsuba_differences.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Difference heatmaps: {output_path.name}")

def generate_relative_displacement_analysis(results, output_dir, pattern_size, file_paths):
    """
    Generate heatmaps showing displacement relative to "no window" baseline
    
    This shows how much the window (undeformed and deformed) changes the 
    displacement pattern compared to the baseline (no window) case.
    
    Parameters:
    -----------
    results : dict
        Dictionary of ComparisonResult objects from analyze_all_cases
    output_dir : Path
        Directory to save output images
    pattern_size : tuple
        (width, height) of checkerboard inner corners
    file_paths : dict
        Dictionary of file paths to load individual engine images
    """
    
    output_dir = Path(output_dir)
    
    if 'no_window' not in results:
        print("  Cannot generate relative displacement - no baseline available")
        return
    
    print("\n  Generating relative displacement analysis...")
    
    cases_to_compare = [
        ('undeformed', 'Undeformed Window'),
        ('deformed', 'Deformed Window')
    ]
    
    engines = ['blender', 'mitsuba']
    
    # Create 2 rows (cases) × 3 columns (Blender, Mitsuba, Difference)
    fig, axes = plt.subplots(len(cases_to_compare), 3, 
                             figsize=(18, 6*len(cases_to_compare)))
    
    if len(cases_to_compare) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Relative Displacement from "No Window" Baseline', 
                 fontsize=16, fontweight='bold')

    
    # Load baseline corners for both engines
    baseline_corners = {}
    for engine in engines:
        baseline_path = Path(file_paths[engine]['no_window'])
        if not baseline_path.exists():
            print(f"    Baseline not found for {engine}")
            continue
        
        baseline_img = cv2.imread(str(baseline_path), cv2.IMREAD_GRAYSCALE)
        if baseline_img is None:
            print(f"    Failed to load baseline for {engine}")
            continue
        
        ret, corners = find_corners_robust(baseline_img, pattern_size, 
                                          f"{engine}_baseline", verbose=False)
        
        if ret:
            baseline_corners[engine] = corners
            print(f"    Loaded baseline for {engine}")
        else:
            print(f"    Failed to find corners in baseline for {engine}")
    
    # Store relative displacements for difference calculation
    relative_displacements = {}
    
    # Process each case (row)
    for row, (case_id, case_name) in enumerate(cases_to_compare):
        relative_displacements[case_id] = {}
        
        # Process each engine (columns 0 and 1)
        for col, engine in enumerate(engines):
            ax = axes[row, col]
            engine_title = engine.capitalize()
            
            if engine not in baseline_corners:
                ax.text(0.5, 0.5, f'{engine_title}\nBaseline Missing',
                       ha='center', va='center', fontsize=14,
                       transform=ax.transAxes)
                ax.set_title(f'{engine_title}: Baseline Missing')
                continue
            
            baseline = baseline_corners[engine]
            case_path = Path(file_paths[engine][case_id])
            
            if not case_path.exists():
                ax.text(0.5, 0.5, f'Image\nNot Found',
                       ha='center', va='center', fontsize=14,
                       transform=ax.transAxes)
                ax.set_title(f'{engine_title}: {case_name} (Missing)')
                continue
            
            case_img = cv2.imread(str(case_path), cv2.IMREAD_GRAYSCALE)
            if case_img is None:
                ax.text(0.5, 0.5, f'Failed to\nLoad Image',
                       ha='center', va='center', fontsize=14,
                       transform=ax.transAxes)
                ax.set_title(f'{engine_title}: {case_name} (Load Error)')
                continue
            
            ret, case_corners = find_corners_robust(case_img, pattern_size,
                                                   f"{engine}_{case_id}", verbose=False)
            
            if not ret:
                ax.text(0.5, 0.5, f'Corner Detection\nFailed',
                       ha='center', va='center', fontsize=14,
                       transform=ax.transAxes)
                ax.set_title(f'{engine_title}: {case_name} (Detection Failed)')
                continue
            
            relative_displacement = case_corners - baseline
            relative_magnitude = np.linalg.norm(relative_displacement, axis=2).flatten()
            
            # Store for difference calculation
            relative_displacements[case_id][engine] = relative_displacement
            
            displacement_grid = relative_magnitude.reshape(pattern_size[1], pattern_size[0])
            
            im = ax.imshow(displacement_grid, cmap='hot', interpolation='nearest')
            cbar = plt.colorbar(im, ax=ax, label='Relative Displacement (px)')
            
            mean_disp = np.mean(relative_magnitude)
            max_disp = np.max(relative_magnitude)
            
            ax.set_title(f'{engine_title}: {case_name}\n'
                        f'Mean: {mean_disp:.2f}px, Max: {max_disp:.2f}px',
                        fontsize=11)
            ax.set_xlabel('X corner index')
            ax.set_ylabel('Y corner index')
            
            print(f"    ✓ {engine_title} {case_name}: mean={mean_disp:.2f}px, max={max_disp:.2f}px")
        
        # Plot difference (Mitsuba - Blender) in column 2
        ax = axes[row, 2]
        
        # Check if we have data for both engines
        if ('blender' not in relative_displacements[case_id] or
            'mitsuba' not in relative_displacements[case_id]):
            
            ax.text(0.5, 0.5, f'Missing Data\nfor Difference',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)
            ax.set_title(f'Difference: {case_name} (Incomplete)')
            continue
        
        # Calculate difference: Mitsuba - Blender
        diff = (relative_displacements[case_id]['mitsuba'] - 
                relative_displacements[case_id]['blender'])
        diff_magnitude = np.linalg.norm(diff, axis=2).flatten()
        
        diff_grid = diff_magnitude.reshape(pattern_size[1], pattern_size[0])
        
        # Use a diverging colormap for differences
        im = ax.imshow(diff_grid, cmap='RdBu_r', interpolation='nearest')
        cbar = plt.colorbar(im, ax=ax, label='Difference (px)')
        
        mean_diff = np.mean(diff_magnitude)
        max_diff = np.max(diff_magnitude)
        
        ax.set_title(f'Mitsuba - Blender: {case_name}\n'
                    f'Mean: {mean_diff:.2f}px, Max: {max_diff:.2f}px',
                    fontsize=11)
        ax.set_xlabel('X corner index')
        ax.set_ylabel('Y corner index')
        
        print(f"    ✓ Difference {case_name}: mean={mean_diff:.2f}px, max={max_diff:.2f}px")
    
    plt.tight_layout()
    
    output_path = output_dir / 'relative_displacement_from_baseline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Relative displacement plot: {output_path.name}")

def main():
    """Main execution function"""
    
    print("="*80)
    print(" BLENDER vs MITSUBA MULTI-IMAGE COMPARISON")
    print("="*80)
    print()
    
    print("Configuration:")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Pattern size: {PATTERN_SIZE}")
    print()
    
    print("Checking files...")
    missing = []
    for engine in ['blender', 'mitsuba']:
        for case_id in ['no_window', 'undeformed', 'deformed']:
            filepath = Path(FILE_PATHS[engine][case_id])
            if filepath.exists():
                print(f"  {engine}/{case_id}: {filepath}")
            else:
                print(f"  {engine}/{case_id}: {filepath} NOT FOUND")
                missing.append(f"{engine}/{case_id}")
    
    if missing:
        print(f"\n WARNING: {len(missing)} file(s) missing:")
        for m in missing:
            print(f"    - {m}")
        print("\nContinuing with available files...\n")
    
    # Run analysis
    results = analyze_all_cases(FILE_PATHS, OUTPUT_DIR, PATTERN_SIZE)
    
    if results:
        print("\n" + "="*80)
        print(" ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nGenerated files in: {OUTPUT_DIR}")
        print("  - comparison_summary.txt     : Detailed numerical results")
        print("  - comparison_plots.png       : Bar charts and error progression")
        print("  - displacement_fields.png    : Displacement visualizations")
    else:
        print("\n✗ Analysis failed. Check error messages above.")


if __name__ == "__main__":
    main()