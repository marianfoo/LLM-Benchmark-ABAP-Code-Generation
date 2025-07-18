CLASS ltcl_test_palindrome DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.
  PRIVATE SECTION.
    METHODS: test_empty_string FOR TESTING,
      test_aba FOR TESTING,
      test_aaaaa FOR TESTING,
      test_zbcd FOR TESTING,
      test_xywyx FOR TESTING,
      test_xywyz FOR TESTING,
      test_xywzx FOR TESTING.
ENDCLASS.

CLASS ltcl_test_palindrome IMPLEMENTATION.
  METHOD test_empty_string.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_048=>is_palindrome( '' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_aba.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_048=>is_palindrome( 'aba' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_aaaaa.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_048=>is_palindrome( 'aaaaa' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_zbcd.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_048=>is_palindrome( 'zbcd' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_xywyx.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_048=>is_palindrome( 'xywyx' )
      exp = abap_true ).
  ENDMETHOD.

  METHOD test_xywyz.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_048=>is_palindrome( 'xywyz' )
      exp = abap_false ).
  ENDMETHOD.

  METHOD test_xywzx.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_048=>is_palindrome( 'xywzx' )
      exp = abap_false ).
  ENDMETHOD.
ENDCLASS.