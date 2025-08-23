CLASS z_humaneval_erp_003 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .
  PUBLIC SECTION.
    TYPES: ty_supervisor_ids TYPE STANDARD TABLE OF char20 WITH EMPTY KEY.

    CLASS-METHODS:
      find_management_chain
        IMPORTING employee_id        TYPE char20
        RETURNING VALUE(rt_supervisors) TYPE ty_supervisor_ids.
ENDCLASS.

CLASS z_humaneval_erp_003 IMPLEMENTATION.

  METHOD find_management_chain.
    DATA: lv_current_emp TYPE char20,
          lt_chain       TYPE ty_supervisor_ids,
          lv_supervisor  TYPE char20.

    lv_current_emp = employee_id.

    WHILE lv_current_emp IS NOT INITIAL.
      SELECT SINGLE supervisor_id
        INTO @lv_supervisor
        FROM zorg_structure
        WHERE employee_id = @lv_current_emp.

      IF sy-subrc <> 0 OR lv_supervisor IS INITIAL.
        EXIT.
      ENDIF.

      APPEND lv_supervisor TO lt_chain.
      lv_current_emp = lv_supervisor.
    ENDWHILE.

    rt_supervisors = lt_chain.
  ENDMETHOD.

ENDCLASS.
