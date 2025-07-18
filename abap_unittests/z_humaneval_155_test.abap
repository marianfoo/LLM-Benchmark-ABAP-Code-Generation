CLASS ltcl_test_even_odd_count DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_result,
             even_count TYPE i,
             odd_count  TYPE i,
           END OF ty_result.

    METHODS: test_case_7 FOR TESTING,
      test_case_minus_78 FOR TESTING,
      test_case_3452 FOR TESTING,
      test_case_346211 FOR TESTING,
      test_case_minus_345821 FOR TESTING,
      test_case_minus_2 FOR TESTING,
      test_case_minus_45347 FOR TESTING,
      test_case_0 FOR TESTING.
ENDCLASS.

CLASS ltcl_test_even_odd_count IMPLEMENTATION.
  METHOD test_case_7.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( 7 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 0 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 1 ).
  ENDMETHOD.

  METHOD test_case_minus_78.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( -78 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 1 ).
  ENDMETHOD.

  METHOD test_case_3452.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( 3452 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 2 ).
  ENDMETHOD.

  METHOD test_case_346211.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( 346211 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 3 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 3 ).
  ENDMETHOD.

  METHOD test_case_minus_345821.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( -345821 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 3 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 3 ).
  ENDMETHOD.

  METHOD test_case_minus_2.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( -2 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 0 ).
  ENDMETHOD.

  METHOD test_case_minus_45347.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( -45347 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 3 ).
  ENDMETHOD.

  METHOD test_case_0.
    DATA: lv_result TYPE ty_result.
    lv_result = z_humaneval_155=>even_odd_count( 0 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-even_count exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = lv_result-odd_count exp = 0 ).
  ENDMETHOD.
ENDCLASS.