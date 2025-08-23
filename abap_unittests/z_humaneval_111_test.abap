CLASS ltcl_test_histogram DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_result,
             letter TYPE string,
             count  TYPE i,
           END OF ty_result.
    TYPES: tt_result TYPE TABLE OF ty_result.

    METHODS: test_case_1 FOR TESTING,
      test_case_2 FOR TESTING,
      test_case_3 FOR TESTING,
      test_case_4 FOR TESTING,
      test_case_5 FOR TESTING,
      test_case_6 FOR TESTING,
      test_case_7 FOR TESTING,
      test_case_8 FOR TESTING.

ENDCLASS.

CLASS ltcl_test_histogram IMPLEMENTATION.

  METHOD test_case_1.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = 'a b b a' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 2
    ).

    READ TABLE et_result INTO DATA(ls_result) WITH KEY letter = 'a'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 2
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'b'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 2
    ).
  ENDMETHOD.

  METHOD test_case_2.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = 'a b c a b' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 2
    ).

    READ TABLE et_result INTO DATA(ls_result) WITH KEY letter = 'a'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 2
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'b'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 2
    ).
  ENDMETHOD.

  METHOD test_case_3.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = 'a b c d g' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 5
    ).

    READ TABLE et_result INTO DATA(ls_result) WITH KEY letter = 'a'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'b'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'c'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'd'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'g'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).
  ENDMETHOD.

  METHOD test_case_4.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = 'r t g' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 3
    ).

    READ TABLE et_result INTO DATA(ls_result) WITH KEY letter = 'r'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 't'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'g'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).
  ENDMETHOD.

  METHOD test_case_5.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = 'b b b b a' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 1
    ).

    READ TABLE et_result INTO DATA(ls_result) WITH KEY letter = 'b'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 4
    ).
  ENDMETHOD.

  METHOD test_case_6.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = 'r t g' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 3
    ).

    READ TABLE et_result INTO DATA(ls_result) WITH KEY letter = 'r'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 't'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).

    READ TABLE et_result INTO ls_result WITH KEY letter = 'g'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).
  ENDMETHOD.

  METHOD test_case_7.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = '' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 0
    ).
  ENDMETHOD.

  METHOD test_case_8.
    DATA: et_result TYPE tt_result.

    et_result = z_humaneval_111=>histogram( test = 'a' ).

    cl_abap_unit_assert=>assert_equals(
      act = lines( et_result )
      exp = 1
    ).

    READ TABLE et_result INTO DATA(ls_result) WITH KEY letter = 'a'.
    cl_abap_unit_assert=>assert_equals(
      act = ls_result-count
      exp = 1
    ).
  ENDMETHOD.

ENDCLASS.