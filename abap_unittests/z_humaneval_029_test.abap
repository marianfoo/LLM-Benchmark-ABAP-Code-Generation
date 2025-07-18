CLASS z_humaneval_029_test DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PRIVATE SECTION.
    METHODS test_filter_empty FOR TESTING.
    METHODS test_filter_by_prefix_xxx FOR TESTING.
ENDCLASS.

CLASS z_humaneval_029_test IMPLEMENTATION.

  METHOD test_filter_empty.
    DATA(result) = z_humaneval_029=>filter_by_prefix(
      strings = VALUE string_table( )
      prefix = 'john'
    ).
    cl_aunit_assert=>assert_equals(
      act = result
      exp = VALUE string_table( )
    ).
  ENDMETHOD.

  METHOD test_filter_by_prefix_xxx.
    DATA(test_strings) = VALUE string_table( ).
    APPEND 'xxx' TO test_strings.
    APPEND 'asd' TO test_strings.
    APPEND 'xxy' TO test_strings.
    APPEND 'john doe' TO test_strings.
    APPEND 'xxxAAA' TO test_strings.
    APPEND 'xxx' TO test_strings.

    DATA(result) = z_humaneval_029=>filter_by_prefix(
      strings = test_strings
      prefix = 'xxx'
    ).

    DATA(expected_result) = VALUE string_table( ).
    APPEND 'xxx' TO expected_result.
    APPEND 'xxxAAA' TO expected_result.
    APPEND 'xxx' TO expected_result.

    cl_aunit_assert=>assert_equals(
      act = result
      exp = expected_result
    ).
  ENDMETHOD.

ENDCLASS.