CLASS z_humaneval_012_test DEFINITION
  FOR TESTING
  FINAL
  RISK LEVEL HARMLESS
  CREATE PUBLIC.

  PRIVATE SECTION.
    METHODS: test_empty_table FOR TESTING,
      test_equal_length_strings FOR TESTING,
      test_different_length_strings FOR TESTING.
ENDCLASS.

CLASS z_humaneval_012_test IMPLEMENTATION.

  METHOD test_empty_table.
    DATA: lt_strings TYPE stringtab,
          result     TYPE string.

    result = z_humaneval_012=>longest( lt_strings ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = ''
    ).
  ENDMETHOD.

  METHOD test_equal_length_strings.
    DATA: lt_strings TYPE stringtab,
          result     TYPE string.

    CLEAR lt_strings.
    APPEND 'x' TO lt_strings.
    APPEND 'y' TO lt_strings.
    APPEND 'z' TO lt_strings.

    result = z_humaneval_012=>longest( lt_strings ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = 'x'
    ).
  ENDMETHOD.

  METHOD test_different_length_strings.
    DATA: lt_strings TYPE stringtab,
          result     TYPE string.

    CLEAR lt_strings.
    APPEND 'x' TO lt_strings.
    APPEND 'yyy' TO lt_strings.
    APPEND 'zzzz' TO lt_strings.
    APPEND 'www' TO lt_strings.
    APPEND 'kkkk' TO lt_strings.
    APPEND 'abc' TO lt_strings.

    result = z_humaneval_012=>longest( lt_strings ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = 'zzzz'
    ).
  ENDMETHOD.

ENDCLASS.