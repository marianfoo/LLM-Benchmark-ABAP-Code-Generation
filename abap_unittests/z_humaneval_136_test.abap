CLASS ltcl_test_z_humaneval_136 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_result,
             largest_negative       TYPE i,
             smallest_positive      TYPE i,
             largest_negative_null  TYPE abap_bool,
             smallest_positive_null TYPE abap_bool,
           END OF ty_result.

    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING,
      test_case_9 FOR TESTING,
      test_case_10 FOR TESTING,
      test_case_11 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_z_humaneval_136 IMPLEMENTATION.

  METHOD test_case_1.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( 2 ) ( 4 ) ( 1 ) ( 3 ) ( 5 ) ( 7 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_true ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( 2 ) ( 4 ) ( 1 ) ( 3 ) ( 5 ) ( 7 ) ( 0 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_true ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( 1 ) ( 3 ) ( 2 ) ( 4 ) ( 5 ) ( 6 ) ( -2 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative exp = -2 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_false ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( 4 ) ( 5 ) ( 3 ) ( 6 ) ( 2 ) ( 7 ) ( -7 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative exp = -7 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_false ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( 7 ) ( 3 ) ( 8 ) ( 4 ) ( 9 ) ( 2 ) ( 5 ) ( -9 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative exp = -9 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_false ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive exp = 2 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_true ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( 0 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_true ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( -1 ) ( -3 ) ( -5 ) ( -6 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative exp = -1 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_false ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_9.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( -1 ) ( -3 ) ( -5 ) ( -6 ) ( 0 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative exp = -1 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_false ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_true ).
  ENDMETHOD.

  METHOD test_case_10.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( -6 ) ( -4 ) ( -4 ) ( -3 ) ( 1 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative exp = -3 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_false ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_false ).
  ENDMETHOD.

  METHOD test_case_11.
    DATA: lt_input  TYPE int4_table,
          ls_result TYPE ty_result.

    lt_input = VALUE #( ( -6 ) ( -4 ) ( -4 ) ( -3 ) ( -100 ) ( 1 ) ).
    ls_result = z_humaneval_136=>largest_smallest_integers( lt_input ).

    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative exp = -3 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-largest_negative_null exp = abap_false ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive exp = 1 ).
    cl_abap_unit_assert=>assert_equals( act = ls_result-smallest_positive_null exp = abap_false ).
  ENDMETHOD.

ENDCLASS.