CLASS zcl_test_humaneval_083 DEFINITION FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_check_input_1 FOR TESTING,
      test_check_input_2 FOR TESTING,
      test_check_input_3 FOR TESTING,
      test_check_input_4 FOR TESTING,
      test_check_input_5 FOR TESTING.
ENDCLASS.

CLASS zcl_test_humaneval_083 IMPLEMENTATION.

  METHOD test_check_input_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_083=>check( 1 )
      exp = 1
      msg = 'Failed for input 1'
    ).
  ENDMETHOD.

  METHOD test_check_input_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_083=>check( 2 )
      exp = 18
      msg = 'Failed for input 2'
    ).
  ENDMETHOD.

  METHOD test_check_input_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_083=>check( 3 )
      exp = 180
      msg = 'Failed for input 3'
    ).
  ENDMETHOD.

  METHOD test_check_input_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_083=>check( 4 )
      exp = 1800
      msg = 'Failed for input 4'
    ).
  ENDMETHOD.

  METHOD test_check_input_5.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_083=>check( 5 )
      exp = 18000
      msg = 'Failed for input 5'
    ).
  ENDMETHOD.

ENDCLASS.