#-------------------------------------------------------------------------
# pyvale: simple,2Dplate,1mat,thermomechanical,steady
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

# Mechanical Props: OFHC Copper 250degC
mgf2EMod = 138e9      # Pa (between 138 and 141.9)
mgf2PRatio = 0.271     # Change between 0.271 and 0.276 (values found in literature)


#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------


[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]


[Mesh]
  [./circle_base]
    type = ConcentricCircleMeshGenerator
    radii = '89e-3'
    rings = '20' 
    num_sectors = 20 
    has_outer_square = false 
    preserve_volumes = false
  []
  [./cylinder_mesh]
    type = MeshExtruderGenerator
    input = circle_base 
    extrusion_vector = '0 0 6.5e-3'
    num_layers = 20
    bottom_sideset = 'bottom'
    top_sideset = 'top'
  []
[]


[Physics/SolidMechanics/QuasiStatic]
  [all]
    strain = FINITE
    add_variables = true
    spherical_center_point = '44.5e-3 44.5e-3 3.25e-3'
    generate_output = 'radial_stress'
  []
[]

[Materials]
    [magnesium_flouride_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${mgf2EMod}
        poissons_ratio = ${mgf2PRatio}
    []
    [stress]
        type = ComputeFiniteStrainElasticStress # ComputeLinearElasticStress or ComputeFiniteStrainElasticStress
    []
[]

[BCs]
  [./fix_bottom_x]
    type = DirichletBC
    variable = disp_x
    boundary = 'outer'
    value = 0.0
  [../]
  [./fix_bottom_y]
    type = DirichletBC
    variable = disp_y
    boundary = 'outer'
    value = 0.0
  [../]
  [./fix_bottom_z]
    type = DirichletBC
    variable = disp_z
    boundary = 'outer'
    value = 0.0
  [../]
  [./pressure_top]
    type = Pressure
    variable = disp_z
    boundary = 'top'
    factor = -101325.0       # pressure in Pascals (negative for downward)
    use_displaced_mesh = true
  [../]
[]


[Preconditioning]
    [smp]
        type = SMP
        full = true
    []
[]

[Executioner]
  type = Steady
  solve_type = PJFNK
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre    boomeramg'
[]

[Outputs]
    exodus = true
[]