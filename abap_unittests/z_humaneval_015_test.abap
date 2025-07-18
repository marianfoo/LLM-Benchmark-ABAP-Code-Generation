CLASS Z_HUMANEVAL_015_test DEFINITION
  FOR TESTING
  FINAL
  RISK LEVEL HARMLESS
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS:
      test_sequence_zero FOR TESTING,
      test_sequence_three FOR TESTING,
      test_sequence_ten FOR TESTING.

  PRIVATE SECTION.
ENDCLASS.

CLASS Z_HUMANEVAL_015_test IMPLEMENTATION.

  METHOD test_sequence_zero.
    DATA(lv_result) = z_humaneval_015=>string_sequence( 0 ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '0' ).
  ENDMETHOD.

  METHOD test_sequence_three.
    DATA(lv_result) = z_humaneval_015=>string_sequence( 3 ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '0 1 2 3' ).
  ENDMETHOD.

  METHOD test_sequence_ten.
    DATA(lv_result) = z_humaneval_015=>string_sequence( 10 ).
    cl_abap_unit_assert=>assert_equals(
      act = lv_result
      exp = '0 1 2 3 4 5 6 7 8 9 10' ).
  ENDMETHOD.

ENDCLASS.