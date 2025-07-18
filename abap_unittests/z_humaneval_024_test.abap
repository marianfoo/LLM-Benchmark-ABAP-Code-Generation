CLASS z_humaneval_024_demo DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS test_largest_divisor_3 FOR TESTING.
    METHODS test_largest_divisor_7 FOR TESTING.
    METHODS test_largest_divisor_10 FOR TESTING.
    METHODS test_largest_divisor_100 FOR TESTING.
    METHODS test_largest_divisor_49 FOR TESTING.
ENDCLASS.

CLASS z_humaneval_024_demo IMPLEMENTATION.

  METHOD test_largest_divisor_3.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_024=>largest_divisor( 3 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 1 ).
  ENDMETHOD.

  METHOD test_largest_divisor_7.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_024=>largest_divisor( 7 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 1 ).
  ENDMETHOD.

  METHOD test_largest_divisor_10.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_024=>largest_divisor( 10 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 5 ).
  ENDMETHOD.

  METHOD test_largest_divisor_100.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_024=>largest_divisor( 100 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 50 ).
  ENDMETHOD.

  METHOD test_largest_divisor_49.
    DATA lv_result TYPE i.
    lv_result = z_humaneval_024=>largest_divisor( 49 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result exp = 7 ).
  ENDMETHOD.

ENDCLASS.