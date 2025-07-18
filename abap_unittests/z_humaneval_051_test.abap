CLASS ltcl_test_remove_vowels DEFINITION FOR TESTING DURATION SHORT RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_empty_string FOR TESTING,
      test_string_with_newline FOR TESTING,
      test_reverse_alphabet FOR TESTING,
      test_only_vowels FOR TESTING,
      test_mixed_case_1 FOR TESTING,
      test_mixed_case_2 FOR TESTING,
      test_no_vowels FOR TESTING.

ENDCLASS.

CLASS ltcl_test_remove_vowels IMPLEMENTATION.

  METHOD test_empty_string.
    DATA: result TYPE string.
    result = z_humaneval_051=>remove_vowels( '' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '' ).
  ENDMETHOD.

  METHOD test_string_with_newline.
    DATA: result TYPE string.
    result = z_humaneval_051=>remove_vowels( |abcdef{ cl_abap_char_utilities=>newline }ghijklm| ).
    cl_abap_unit_assert=>assert_equals( act = result exp = |bcdf{ cl_abap_char_utilities=>newline }ghjklm| ).
  ENDMETHOD.

  METHOD test_reverse_alphabet.
    DATA: result TYPE string.
    result = z_humaneval_051=>remove_vowels( 'fedcba' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'fdcb' ).
  ENDMETHOD.

  METHOD test_only_vowels.
    DATA: result TYPE string.
    result = z_humaneval_051=>remove_vowels( 'eeeee' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = '' ).
  ENDMETHOD.

  METHOD test_mixed_case_1.
    DATA: result TYPE string.
    result = z_humaneval_051=>remove_vowels( 'acBAA' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'cB' ).
  ENDMETHOD.

  METHOD test_mixed_case_2.
    DATA: result TYPE string.
    result = z_humaneval_051=>remove_vowels( 'EcBOO' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'cB' ).
  ENDMETHOD.

  METHOD test_no_vowels.
    DATA: result TYPE string.
    result = z_humaneval_051=>remove_vowels( 'ybcd' ).
    cl_abap_unit_assert=>assert_equals( act = result exp = 'ybcd' ).
  ENDMETHOD.

ENDCLASS.