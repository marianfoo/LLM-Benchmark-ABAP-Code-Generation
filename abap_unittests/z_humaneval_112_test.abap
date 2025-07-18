CLASS ltc_test_reverse_delete DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    TYPES: BEGIN OF ty_result,
             result_string TYPE string,
             is_palindrome TYPE abap_bool,
           END OF ty_result.

    METHODS: test_abcde_ae FOR TESTING,
      test_abcdef_b FOR TESTING,
      test_abcdedcba_ab FOR TESTING,
      test_dwik_w FOR TESTING,
      test_a_a FOR TESTING,
      test_abcdedcba_empty FOR TESTING,
      test_abcdedcba_v FOR TESTING,
      test_vabba_v FOR TESTING,
      test_mamma_mia FOR TESTING.

ENDCLASS.

CLASS ltc_test_reverse_delete IMPLEMENTATION.

  METHOD test_abcde_ae.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'abcde' c = 'ae' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = 'bcd' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_false ).
  ENDMETHOD.

  METHOD test_abcdef_b.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'abcdef' c = 'b' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = 'acdef' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_false ).
  ENDMETHOD.

  METHOD test_abcdedcba_ab.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'abcdedcba' c = 'ab' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = 'cdedc' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_true ).
  ENDMETHOD.

  METHOD test_dwik_w.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'dwik' c = 'w' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = 'dik' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_false ).
  ENDMETHOD.

  METHOD test_a_a.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'a' c = 'a' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = '' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_true ).
  ENDMETHOD.

  METHOD test_abcdedcba_empty.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'abcdedcba' c = '' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = 'abcdedcba' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_true ).
  ENDMETHOD.

  METHOD test_abcdedcba_v.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'abcdedcba' c = 'v' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = 'abcdedcba' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_true ).
  ENDMETHOD.

  METHOD test_vabba_v.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'vabba' c = 'v' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = 'abba' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_true ).
  ENDMETHOD.

  METHOD test_mamma_mia.
    DATA: result TYPE ty_result.
    result = z_humaneval_112=>reverse_delete( s = 'mamma' c = 'mia' ).
    cl_abap_unit_assert=>assert_equals( act = result-result_string exp = '' ).
    cl_abap_unit_assert=>assert_equals( act = result-is_palindrome exp = abap_true ).
  ENDMETHOD.

ENDCLASS.