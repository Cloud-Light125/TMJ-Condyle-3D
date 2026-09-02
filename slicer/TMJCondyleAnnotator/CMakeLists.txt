set(MODULE_NAME TMJCondyleAnnotator)

set(MODULE_PYTHON_FILES
  TMJCondyleAnnotator.py
  __init__.py
)

set(MODULE_DOCUMENTATION_FILES
  README.md
)

slicerMacroBuildScriptedModule(
  NAME TMJCondyleAnnotator
  SCRIPTS ${MODULE_PYTHON_FILES}
  DOCUMENTATION ${MODULE_DOCUMENTATION_FILES}
)
