CLASS ltcl_test_circular_shift DEFINITION FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_circular_shift IMPLEMENTATION.
  METHOD test_case_1.
    DATA: result TYPE string.
    result = z_humaneval_065=>circular_shift( x = 100 shift = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '001' ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: result TYPE string.
    result = z_humaneval_065=>circular_shift( x = 12 shift = 2 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '12' ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: result TYPE string.
    result = z_humaneval_065=>circular_shift( x = 97 shift = 8 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '79' ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: result TYPE string.
    result = z_humaneval_065=>circular_shift( x = 12 shift = 1 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '21' ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: result TYPE string.
    result = z_humaneval_065=>circular_shift( x = 11 shift = 101 ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '11' ).
  ENDMETHOD.
ENDCLASS.