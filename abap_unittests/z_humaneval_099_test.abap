CLASS ltcl_test_closest_integer DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS:
      test_closest_integer_case_1 FOR TESTING,
      test_closest_integer_case_2 FOR TESTING,
      test_closest_integer_case_3 FOR TESTING,
      test_closest_integer_case_4 FOR TESTING,
      test_closest_integer_case_0 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_closest_integer IMPLEMENTATION.

  METHOD test_closest_integer_case_1.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_099=>closest_integer( '10' )
      exp = 10
      msg = 'Test 1' ).
  ENDMETHOD.

  METHOD test_closest_integer_case_2.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_099=>closest_integer( '14.5' )
      exp = 15
      msg = 'Test 2' ).
  ENDMETHOD.

  METHOD test_closest_integer_case_3.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_099=>closest_integer( '-15.5' )
      exp = -16
      msg = 'Test 3' ).
  ENDMETHOD.

  METHOD test_closest_integer_case_4.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_099=>closest_integer( '15.3' )
      exp = 15
      msg = 'Test 4' ).
  ENDMETHOD.

  METHOD test_closest_integer_case_0.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_099=>closest_integer( '0' )
      exp = 0
      msg = 'Test 0' ).
  ENDMETHOD.

ENDCLASS.