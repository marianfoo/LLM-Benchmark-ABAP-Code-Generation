CLASS zcl_HUMANEVAL_084 DEFINITION FOR TESTING
DURATION SHORT
RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_solve_1000 FOR TESTING,
      test_solve_150 FOR TESTING,
      test_solve_147 FOR TESTING,
      test_solve_333 FOR TESTING,
      test_solve_963 FOR TESTING.
ENDCLASS.

CLASS zcl_HUMANEVAL_084 IMPLEMENTATION.

  METHOD test_solve_1000.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_084=>solve( 1000 )
      exp = '1'
      msg = 'Error testing solve with input 1000' ).
  ENDMETHOD.

  METHOD test_solve_150.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_084=>solve( 150 )
      exp = '110'
      msg = 'Error testing solve with input 150' ).
  ENDMETHOD.

  METHOD test_solve_147.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_084=>solve( 147 )
      exp = '1100'
      msg = 'Error testing solve with input 147' ).
  ENDMETHOD.

  METHOD test_solve_333.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_084=>solve( 333 )
      exp = '1001'
      msg = 'Error testing solve with input 333' ).
  ENDMETHOD.

  METHOD test_solve_963.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_084=>solve( 963 )
      exp = '10010'
      msg = 'Error testing solve with input 963' ).
  ENDMETHOD.

ENDCLASS.