CLASS ltcl_z_humaneval_erp_004 DEFINITION FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.

    METHODS:
      setup,
      teardown,

      test_single_supervisor FOR TESTING,
      test_multiple_supervisors FOR TESTING,
      test_no_supervisor FOR TESTING.

ENDCLASS.

CLASS ltcl_z_humaneval_erp_004 IMPLEMENTATION.

  METHOD setup.
    DATA ls_entry TYPE zorg_structure.

    " Insert chain: A -> B -> C
    ls_entry-employee_id = 'A'.
    ls_entry-supervisor_id = 'B'.
    INSERT zorg_structure FROM ls_entry.

    ls_entry-employee_id = 'B'.
    ls_entry-supervisor_id = 'C'.
    INSERT zorg_structure FROM ls_entry.

    " 'C' has no supervisor, so no entry for C
  ENDMETHOD.

  METHOD teardown.
    DELETE FROM zorg_structure WHERE employee_id IN ( 'A', 'B' ).
  ENDMETHOD.

  METHOD test_single_supervisor.
    DATA(result) = z_humaneval_erp_003=>find_management_chain( employee_id = 'B' ).
    cl_abap_unit_assert=>assert_equals(
      act = lines( result )
      exp = 1
      msg = 'Supervisor chain length for B should be 1'
    ).
    cl_abap_unit_assert=>assert_equals(
      act = result[ 1 ]
      exp = 'C'
      msg = 'B should have C as supervisor'
    ).
  ENDMETHOD.

  METHOD test_multiple_supervisors.
    DATA(result) = z_humaneval_erp_003=>find_management_chain( employee_id = 'A' ).
    cl_abap_unit_assert=>assert_equals(
      act = lines( result )
      exp = 2
      msg = 'Supervisor chain length for A should be 2'
    ).
    cl_abap_unit_assert=>assert_equals(
      act = result[ 1 ]
      exp = 'B'
      msg = 'First supervisor of A should be B'
    ).
    cl_abap_unit_assert=>assert_equals(
      act = result[ 2 ]
      exp = 'C'
      msg = 'Second supervisor of A should be C'
    ).
  ENDMETHOD.

  METHOD test_no_supervisor.
    DATA(result) = z_humaneval_erp_003=>find_management_chain( employee_id = 'C' ).
    cl_abap_unit_assert=>assert_equals(
      act = lines( result )
      exp = 0
      msg = 'C has no supervisor, result should be empty'
    ).
  ENDMETHOD.

ENDCLASS.