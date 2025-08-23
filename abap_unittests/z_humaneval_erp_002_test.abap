CLASS ltcl_test_humaneval_erp_003 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      setup,
      test_no_subordinates FOR TESTING,
      test_multiple_subordinates FOR TESTING,
      test_one_subordinate FOR TESTING.

ENDCLASS.

CLASS ltcl_test_humaneval_erp_003 IMPLEMENTATION.

  METHOD setup.
    DELETE FROM zemp_master.
    DELETE FROM zorg_structure.
    COMMIT WORK.
  ENDMETHOD.


  METHOD test_no_subordinates.

    DATA: ls_entry TYPE zorg_structure.

    ls_entry-employee_id = 'A'.
    ls_entry-supervisor_id = 'B'.
    INSERT zorg_structure FROM ls_entry.

    ls_entry-employee_id = 'C'.
    ls_entry-supervisor_id = 'D'.
    INSERT zorg_structure FROM ls_entry.

    DATA(lv_result) = z_humaneval_erp_002=>count_subordinates( supervisor_id = 'X' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 0
      msg = 'X has no subordinates'
    ).
  ENDMETHOD.

  METHOD test_multiple_subordinates.

    DATA: ls_entry TYPE zorg_structure.

    ls_entry-employee_id = 'A'.
    ls_entry-supervisor_id = 'Z'.
    INSERT zorg_structure FROM ls_entry.

    ls_entry-employee_id = 'B'.
    ls_entry-supervisor_id = 'Z'.
    INSERT zorg_structure FROM ls_entry.

    ls_entry-employee_id = 'C'.
    ls_entry-supervisor_id = 'Z'.
    INSERT zorg_structure FROM ls_entry.

    ls_entry-employee_id = 'D'.
    ls_entry-supervisor_id = 'Y'.
    INSERT zorg_structure FROM ls_entry.

    DATA(lv_result) = z_humaneval_erp_002=>count_subordinates( supervisor_id = 'Z' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 3
      msg = 'Z should have 3 subordinates'
    ).
  ENDMETHOD.

  METHOD test_one_subordinate.

    DATA: ls_entry TYPE zorg_structure.

    ls_entry-employee_id = 'A'.
    ls_entry-supervisor_id = 'M'.
    INSERT zorg_structure FROM ls_entry.

    ls_entry-employee_id = 'B'.
    ls_entry-supervisor_id = 'N'.
    INSERT zorg_structure FROM ls_entry.

    DATA(lv_result) = z_humaneval_erp_002=>count_subordinates( supervisor_id = 'M' ).

    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = 1
      msg = 'M should have 1 subordinate'
    ).
  ENDMETHOD.

ENDCLASS.